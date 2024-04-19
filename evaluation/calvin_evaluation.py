# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for evaluating GR-1 on Calvin Benchmark."""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image

import clip

import models.vision_transformer as vits
from models.gr1 import GR1

from calvin_agent.models.calvin_base_model import CalvinBaseModel


class GR1CalvinEvaluation(CalvinBaseModel):
    def __init__(self,
                 mae_ckpt,
                 policy_ckpt,
                 variant,
                 device
    ):
        """Constructor."""
        self.tokenizer = clip.tokenize
        self.variant = variant
        self.seq_len = variant['seq_len']
        self.use_hand_rgb = variant['use_hand_rgb']
        self.act_dim = variant['act_dim']
        self.state_dim = variant['state_dim']
        self.device = device

        # Preprocess
        input_size = (224, 224)
        rgb_mean = (0.485, 0.456, 0.406)
        rgb_std = (0.229, 0.224, 0.225)
        self.preprocess = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(rgb_mean, rgb_std)])

        # CLIP
        model_clip, _ = clip.load(variant['clip_backbone'], device=self.device)
        
        # MAE
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        model_mae.to(self.device)
        checkpoint = torch.load(mae_ckpt, map_location='cpu')
        model_mae.load_state_dict(checkpoint['model'], strict=False)

        # Resampler hparams
        resampler_params = dict()
        resampler_params['depth'] = variant['resampler_depth']
        resampler_params['dim_head'] = variant['resampler_dim_head']
        resampler_params['heads'] = variant['resampler_heads']
        resampler_params['num_latents'] = variant['resampler_num_latents']
        resampler_params['num_media_embeds'] = variant['resampler_num_media_embeds']
        variant['resampler_params'] = resampler_params

        # GR-1 policy
        self.policy = GR1(
            model_clip=model_clip,
            model_mae=model_mae,
            state_dim=variant['state_dim'],
            act_dim=variant['act_dim'],
            hidden_size=variant['embed_dim'],
            sequence_length=variant['seq_len'],
            training_target=['act_pred'],
            img_feat_dim=variant['img_feat_dim'],
            lang_feat_dim=variant['lang_feat_dim'],
            patch_feat_dim=variant['patch_feat_dim'],
            resampler_params=variant['resampler_params'],
            without_norm_pix_loss=variant['without_norm_pix_loss'],
            use_hand_rgb=variant['use_hand_rgb'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=variant['n_positions'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'])
        print(f"loading state dict: {policy_ckpt}...")
        payload = torch.load(policy_ckpt)
        state_dict = payload['state_dict']
        msg = self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(self.device)
        self.policy.eval()

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """Step function."""
        # Language
        text = goal
        tokenized_text = self.tokenizer(text)

        # RGB
        rgb = obs['rgb_obs']['rgb_static']
        rgb = Image.fromarray(rgb)
        rgb = T.ToTensor()(rgb.convert("RGB"))
        rgb = self.preprocess(rgb)
        self.rgb_list.append(rgb)
        hand_rgb = obs['rgb_obs']['rgb_gripper']
        hand_rgb = Image.fromarray(hand_rgb)
        hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
        hand_rgb = self.preprocess(hand_rgb)
        self.hand_rgb_list.append(hand_rgb)

        # State
        state = obs['robot_obs']
        arm_state = state[:6]
        gripper_state = state[-1]
        state = torch.from_numpy(np.hstack([arm_state, gripper_state]))
        self.state_list.append(state)
        
        # Buffer
        buffer_len = len(self.rgb_list)
        if buffer_len > self.seq_len:
            self.rgb_list.pop(0)
            self.hand_rgb_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_list) == self.seq_len
            assert len(self.hand_rgb_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_list)
        
        # Static RGB
        c, h, w = rgb.shape
        rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_tensor = torch.stack(self.rgb_list, dim=0)  # (l, c, h, w)
        rgb_data[0, :buffer_len] = rgb_tensor

        # Hand RGB
        c, h, w = hand_rgb.shape
        hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0)  # (l, c, h, w)
        hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # State
        state_tensor = torch.stack(self.state_list, dim=0)  # (l, act_dim)
        gripper_state_data = - torch.ones((1, self.seq_len)).float()
        gripper_state_data[0, :buffer_len] = state_tensor[:, 6]
        gripper_state_data = (gripper_state_data + 1.0) / 2
        gripper_state_data = gripper_state_data.long()
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).float()  # (b, l, 2)
        arm_state_data = torch.zeros((1, self.seq_len, self.act_dim - 1)).float()  # (b, l, act_dim - 1)
        arm_state_data[0, :buffer_len] = state_tensor[:, :6]

        # Attention mask
        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1

        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        rgb_data = rgb_data.to(self.device)
        hand_rgb_data = hand_rgb_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                hand_rgb=hand_rgb_data,
                state=state_data,
                language=tokenized_text,
                attention_mask=attention_mask
        )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, l, act_dim - 1)
        arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (l, act_dim - 1)
        arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]

        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, l, 1)
        gripper_action_preds = gripper_action_preds.flatten()  # (l, )
        gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]

        # Use the last action
        arm_action_pred = arm_action_preds[-1]  # (act_dim - 1, )
        gripper_action_pred = gripper_action_preds[-1:]  # (1, )
        gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        gripper_action_pred = gripper_action_pred > 0.5
        gripper_action_pred = gripper_action_pred.int().float()
        gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=0)  # (act_dim,)
        action_pred = action_pred.detach().cpu()

        self.rollout_step_counter += 1
    
        return action_pred
