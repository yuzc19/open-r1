# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""
import re
import os
from typing import Callable
from bert_score import BERTScorer
from transformers import AutoTokenizer
from infer.simple_dataman import DataManInference
from infer.simple_structure import StructureInference
from infer.influence_server import DataInfluenceClient

# fasttext_path = "/project/flame/zichunyu/code/dclm/baselines/mappers/enrichers/quality_prediction_enrichment_models/fasttext_oh_eli5.bin"
# fasttext_model = fasttext.load_model(fasttext_path)
bert_scorer = BERTScorer(
    model_type="microsoft/deberta-large-mnli",
    device=f"cuda:{os.environ.get('LOCAL_RANK', 0)}",
)
# tokenizer = AutoTokenizer.from_pretrained(
#     "bert-base-uncased",
#     max_length=2048,
#     padding="max_length",
# )
dataman_llm = DataManInference(use_server=True)
structure_llm = StructureInference(use_server=True)


def dataman_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    dataman_score: list[float],
    **kwargs,
) -> list[float]:
    """Reward function that uses a data quality model to score completions."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    results = dataman_llm.score_texts(contents)
    dataman_rewards = [r.get("overall_score", 0) for r in results]
    # return [dataman_rewards[i] - dataman_rewards[len(contents)] for i in range(len(contents))]
    return [dataman_rewards[i] - dataman_score[i] for i in range(len(contents))]


def bert_score_reward(
    completions: list[list[dict[str, str]]], text: list[str], **kwargs
) -> list[float]:
    """Reward function that uses BERTScore to compare generated content with original text."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        # match = re.search(r"<answer>\n(.*?)\n</answer>", content, re.DOTALL)
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")
    # print(len(completions), "completions")
    # print(f"MODEL GENERATION:\n{completions[0][0]['content']}")
    # print("-" * 80)
    # print(f"TEXT:\n{text[0]}")
    # print("-" * 80)
    P, R, F1 = bert_scorer.score(contents, text, batch_size=1)

    # Return F1 scores as rewards
    # return F1.tolist()
    return [int(float(f1) > 0.65) for f1 in F1.tolist()]
    # return [int(float(f1) > 0.75) for f1 in F1.tolist()]


def structure_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    dataman_score: list[float],
    **kwargs,
) -> list[float]:
    """Reward function that uses a structure comparison model to score completions."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    structure_rewards = structure_llm.score_texts(text, contents)
    return structure_rewards


def length_reward(
    completions: list[list[dict[str, str]]], text: list[str], **kwargs
) -> list[float]:
    """Reward function that uses the length of the generated content."""
    contents = []
    for completion in completions:
        content = completion[0]["content"]
        # match = re.search(r"<answer>\n(.*?)\n</answer>", content, re.DOTALL)
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")

    return [len(c) <= 1.25 * len(t) for c, t in zip(contents, text)]


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n\nHere is a paraphrased version:.*?"
    pattern = r"^Here is a paraphrased version:.*?"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    # flag = 1.0
    # if any(match is None for match in matches):
    #     flag = 0.0
    return [1.0 if match else 0.0 for match in matches]
    # return [flag for _ in completion_contents]


data_influence_client = DataInfluenceClient()

def data_influence_reward(
    completions: list[list[dict[str, str]]],
    text: list[str],
    **kwargs,
) -> list[float]:
    """Reward with data influence computed from two checkpoints (loss_before - loss_after)."""
    contents = []
    for completion, t in zip(completions, text):
        content = completion[0]["content"]
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match and (c := match.group(1).strip()):
            if len(c) >= 0.7 * len(t):
                # If the paraphrased version is not too short, use it for scoring.
                contents.append(c)
            else:
                # Otherwise, use the original text.
                contents.append(t)
        else:
            contents.append(t)

    cur_values = data_influence_client.score_texts(contents)
    baseline_values = data_influence_client.score_texts(text)

    return [float(cur) - float(base) for cur, base in zip(cur_values, baseline_values)]


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "format": format_reward,
        "dataman": dataman_reward,
        "bert_score": bert_score_reward,
        "structure": structure_reward,
        "length": length_reward,
        "data_influence": data_influence_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
