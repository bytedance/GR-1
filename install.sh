#!/bin/bash

pip3 install packaging==21.3
pip3 install flamingo_pytorch
pip3 install tensorboard
pip3 install ftfy regex tqdm
pip3 install matplotlib
pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
pip3 install transformers==4.5.1
pip3 install git+https://github.com/openai/CLIP.git

sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf