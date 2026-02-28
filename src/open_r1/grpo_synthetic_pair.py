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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
import zstandard as zstd
import json


logger = logging.getLogger(__name__)

# - Aim to create a condensed but accurate and informative version of the original text, not a simplistic summary.

distill_prompt_2 = """Your task is to read two provided texts and synthesize a new text that captures their shared underlying concepts, structure, and core ideas, following these instructions:

* Identify and extract the common conceptual essence:

  * Focus on ideas, principles, mechanisms, arguments, or reasoning patterns that appear in both texts
  * Abstract away from surface wording, stylistic differences, and specific examples
  * Identify shared themes, assumptions, and logical structures

* Remove content that is not shared:

  * Exclude details, examples, anecdotes, or explanations that appear in only one text
  * Do not merge unrelated content
  * Do not include contradictory or text-specific claims

* Preserve meaningful shared substance:

  * Retain technical terms and key concepts that are central to both texts
  * Maintain the depth and rigor of reasoning where both texts support it
  * Keep shared causal structures, definitions, and conceptual relationships

* Avoid paraphrasing either text directly:

  * Do not copy sentence structure or distinctive phrasing
  * Do not produce a summary of Text A or Text B separately
  * The output must be a new, independent text

* Ensure the output:

  * Is coherent and logically structured
  * Reads as a standalone document
  * Is written in high-quality, clear English
  * Does not introduce assumptions or information not supported by both texts

Here are the texts:

Text A:
{TEXT_A}

Text B:
{TEXT_B}

Task:
After thoroughly analyzing both texts, generate a new text that reflects only their shared conceptual essence, abstracted from surface details.

Start your response immediately with:
"Here is a synthesized text capturing their shared essence:"
and then provide the synthesized text."""


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    def load_local_jsonl_zst(file_path):
        """Load data from a compressed JSONL file and yield examples"""

        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read().decode("utf-8")
                for line in text_stream.strip().split("\n"):
                    if line:
                        yield json.loads(line)

    with open("data/50000_sample_low_score.jsonl", "r") as tf:
        dataset = datasets.Dataset.from_list(
            [
                {
                    "text": json.loads(line)["text"][:7000],
                    "dataman_score": json.loads(line)["dataman_score"],
                }
                for line in tf
                # for item in load_local_jsonl_zst("CC_shard_00000000_processed.jsonl.zst")
            ]
        )
    logger.info(f"Dataset loaded with {len(dataset['train'])} examples.")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = "text"):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append(
            {
                "role": "user",
                "content": distill_prompt_2.format(TEXT=example[prompt_column]),
            }
        )
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = -1  # Unknown in streaming mode
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", "synthetic-data"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
