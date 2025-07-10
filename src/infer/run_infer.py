from src.open_r1.grpo_synthetic import distill_prompt
from vllm import LLM, TokensPrompt, SamplingParams
from pathlib import Path as LocalPath
from google.cloud import storage
from cloudpathlib import S3Path
import zstandard as zstd
from tqdm import tqdm
import datasets
import json
import sys
import re

gstore = storage.Client().bucket("cmu-gpucloud-zichunyu")
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.9,
    max_tokens=2048,
)


def load_jsonl_zst(file_path):
    """Load data from a compressed JSONL file and yield examples"""
    if file_path.startswith("s3://"):
        path = S3Path(file_path)
    else:
        path = LocalPath(file_path)

    with path.open("rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = reader.read().decode("utf-8")
            for line in text_stream.strip().split("\n"):
                if line:
                    yield json.loads(line)


def make_conversation(example, prompt_column: str = "text"):
    prompt = []
    prompt.append(
        {
            "role": "system",
            "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. /no_think",
        }
    )
    prompt.append(
        {
            "role": "user",
            "content": distill_prompt.format(TEXT=example[prompt_column]),
        }
    )
    return {"prompt": prompt}


def transform(llm, conversations):
    outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
    contents = []
    for output in outputs:
        content = output.outputs[0].text
        # Find the first occurrence that matches the pattern in the string
        match = re.search(r"Here is a paraphrased version:(.*)", content, re.DOTALL)
        if match:
            # Extract the response within the tags
            response = match.group(1).strip()
            contents.append(response)
        else:
            contents.append("")
    return contents


def vllm_inference(llm, dataset):
    batch_size = 1024
    transformed_texts = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference..."):
        batch = dataset[i : i + batch_size]["prompt"]
        transformed_batch = transform(llm, batch)
        transformed_texts.extend(transformed_batch)
    return transformed_texts


def main():
    rank = int(sys.argv[1])
    print(f"Received argument: {rank}")

    llm = LLM(model="Qwen/Qwen3-1.7B")  # 2400 min / 1B tokens / 1 GPU
    # 3 hour for one file, 96 files for 5B tokens
    read_template = "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/shard_{}_processed.jsonl.zstd"
    write_template = "data/refinedweb_01_0/Qwen3-1.7B/textfiles/shard_{}_processed.jsonl"
    for i in tqdm(range(rank * 4, (rank + 1) * 4), desc="Processing shards..."):
        read_file_name = read_template.format(str(i).zfill(8))
        print(f"Processing file: {read_file_name}")
        data_list = []
        tids = []
        tid = 0
        for item in load_jsonl_zst(read_file_name):
            text = item["text"]
            for j in range(0, len(text), 7000):
                data_list.append({"text": text[j : j + 7000]})
                tids.append(tid)
            tid += 1
        dataset = datasets.Dataset.from_list(data_list).map(make_conversation)
        transformed_texts = vllm_inference(llm, dataset)
        write_file_name = write_template.format(str(i).zfill(8))
        with gstore.blob(write_file_name).open("w") as f:
            print(f"Writing to file: {write_file_name}")
            prev_tid = 0
            texts = []
            cnt = 0
            for tid, text in zip(tids, transformed_texts):
                if tid == prev_tid:
                    texts.append(text)
                else:
                    cur_text = " ".join(texts)
                    if cur_text == "":
                        cnt += 1
                    f.write(json.dumps({"text": cur_text}) + "\n")
                    prev_tid = tid
                    texts = [text]
            cur_text = " ".join(texts)
            if cur_text == "":
                cnt += 1
            f.write(json.dumps({"text": cur_text}) + "\n")
            print(f"Total {len(transformed_texts)} items, {cnt} invalid items found.")


if __name__ == "__main__":
    main()
