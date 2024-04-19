# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import Counter
import json
import numpy as np


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, eval_result_path, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )
