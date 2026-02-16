import argparse
import os
import random

from datasets import Dataset
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from tqdm import tqdm
from transformers import AutoTokenizer
from .data_utils import get_adapter_func, split_into_batches
from .chunk_utils import execute_meta_operations
from vllm import LLM, SamplingParams

import re
from dataclasses import dataclass
from typing import Optional, TypeVar

import yaml

DataclassT = TypeVar("DataclassT")


# numpy==2.2.6
@dataclass
class GentaskConfig:
    "Data Path"

    data_path: Optional[str] = None

    "Save Configs"
    save_path: Optional[str] = "./data/data_gen"
    save_name: Optional[str] = "test"
    save_interval: Optional[int] = None

    @staticmethod
    def from_yaml(filepath):
        """
        Load the configuration from a YAML file
        Args:
            filepath (str): Path to the YAML file
        """
        with open(filepath, "r") as file:
            config_dict = yaml.safe_load(file)

        # Create a new instance of GentaskConfig and update its variables from the loaded data
        config = GentaskConfig
        for key, value in config_dict.items():
            # if value contains env variable, get the value from the environment
            if value is None or not isinstance(value, str):
                pass
            elif "$" in value:  # xxxx/$env_name/xxxxx or xxx/${env_name}/xxxx
                env_var_pattern = re.compile(
                    r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
                )
                envvar_name = env_var_pattern.search(value).group(
                    1
                ) or env_var_pattern.search(value).group(2)
                envvar_value = os.getenv(envvar_name)
                if envvar_value is None:
                    raise ValueError(f"Environment variable {envvar_name} is not set")
                config_dict[key] = value.replace(
                    "${" + envvar_name + "}", envvar_value
                ).replace("$" + envvar_name, envvar_value)

            if hasattr(config, key):
                setattr(config, key, config_dict[key])

        return config

random.seed(42)


# dummy env constants for multi-gpu & multi-node
NODE_GPUS = int(os.environ.get("NODE_GPUS", 8))
NODE_RANK = int(os.environ.get("NODE_RANK", 0))
CUDA_DEVICE = int(os.environ["CUDA_VISIBLE_DEVICES"])
TOTAL_SPLIT = int(os.environ["TOTAL_SPLIT"])


def trunc_text(text: str, tokenizer, max_token=1500, max_digits=3):
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for idx_line, line in enumerate(lines):
        normalize_line = f"[{idx_line:0{max_digits}d}]{line}"
        line_token_count = len(tokenizer.encode(normalize_line))

        # if cur line can be appended in current chunk
        if current_chunk_token_count + line_token_count <= max_token:
            current_chunk.append(normalize_line)
            current_chunk_token_count += line_token_count
        # if cur line cannot be appended in current chunk
        else:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [normalize_line]
            current_chunk_token_count = line_token_count
            if line_token_count > max_token:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_chunk_token_count = 0

    if len(current_chunk) > 0:
        chunks.append("\n".join(current_chunk))

    return chunks


def merge_chunks(chunk_list, chunk_lengths):
    merged_documents = []
    current_index = 0

    for length in chunk_lengths:
        document_chunks = chunk_list[current_index : current_index + length]
        merged_document = "\n".join(document_chunks)
        merged_documents.append(merged_document)
        current_index += length

    return merged_documents


def main(args):
    # load config
    config = GentaskConfig().from_yaml(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.token_template, use_fast=False)
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.bos_token = base_tokenizer.bos_token
    tokenizer.eos_token = base_tokenizer.eos_token

    # prepare data
    data_reader = JsonlReader(
        data_folder=config.data_path,
        file_progress=True,
        compression="zstd",
        adapter=get_adapter_func(args.dataset_name),
        limit=args.limit,
    )

    arguments = [] 
    for idx, doc in enumerate(
        data_reader.run(
            rank=CUDA_DEVICE + NODE_RANK * NODE_GPUS, world_size=TOTAL_SPLIT
        )
    ):
        arguments.append({"text": doc.text})

    dir_path = os.path.join(
        config.save_path, config.save_name + f"_{CUDA_DEVICE + NODE_RANK * NODE_GPUS}"
    )
    os.makedirs(dir_path, exist_ok=True)
    base_name = config.save_name
    batches = split_into_batches(arguments, config.save_interval)

    tp_size = 1  # Tensor Parallelism
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=256)
    engine = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tp_size,
    )

    for i, batch in enumerate(tqdm(batches)):
        if os.path.exists(
            os.path.join(
                dir_path,
                f"{base_name}_{i + 1}_{(len(arguments) - 1) // config.save_interval + 1}.jsonl",
            )
        ):
            print(
                f"{os.path.join(dir_path, f'{base_name}_{i + 1}_{(len(arguments) - 1) // config.save_interval + 1}.jsonl')} already exists!"
            )
            continue
        rets = []
        chunk_lens = []
        for idx, sample in enumerate(tqdm(batch, total=len(batch), unit="tokenizing")):
            if sample["text"] == "":
                chunk_lens.append(0)
                continue
            user_msgs = trunc_text(sample["text"], base_tokenizer, max_token=1500)
            chunk_lens.append(len(user_msgs))
            for user_msg in user_msgs:
                user_msg = f"[doc]\n{user_msg}\n[/doc]"
                total_msg = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant.",
                        },
                        {"role": "user", "content": user_msg},
                    ],
                    add_generation_prompt=True,
                    # tokenize=False,
                    truncation=True,
                    max_length=1792,
                )
                rets.append(total_msg)
        outputs = engine.generate(prompt_token_ids=rets, sampling_params=sampling_params)
        outputs = [item.outputs[0].text.strip(" ") for item in outputs]
        merged_outputs = merge_chunks(outputs, chunk_lens)
        rets = [
            {
                # "raw_content": sample["metadata"]["raw_content"],
                # "doc_content": sample["text"],
                "text": execute_meta_operations(
                    text=sample["text"],
                    operations=output,
                    threshold_1=args.threshold_1,
                    threshold_2=args.threshold_2,
                    error_op=args.error_op,
                ),
                # "metadata": {
                #     "doc_program": sample["metadata"]["doc_program"],
                #     "chunk_program": output,
                # },
            }
            for sample, output in zip(batch, merged_outputs)
        ]

        intermediate_ds = Dataset.from_list(rets)

        intermediate_ds.to_json(
            os.path.join(
                dir_path,
                f"{base_name}_{i + 1}_{(len(arguments) - 1) // config.save_interval + 1}.jsonl",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--token_template",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument("--batch_size", type=int, default=1000000)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process, for debugging.",
    )
    parser.add_argument("--data_format", type=str, default="parquet")
    parser.add_argument("--threshold_1", type=float, default=0.0)
    parser.add_argument("--threshold_2", type=float, default=0.95)
    parser.add_argument("--error_op", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default="c4")
    args = parser.parse_args()
    main(args)
