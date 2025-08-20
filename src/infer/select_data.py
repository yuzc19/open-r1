import os
import json
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from pathlib import Path
from file_utils import read_jsonl, write_jsonl


def dataman_select(dataset_size, selection_size, args):
    """Select top 10% based on overall_score from DataMan output files."""
    all_scores = []

    # Read scores from all 512 shards
    for i in tqdm(range(512), desc="Loading DataMan scores"):
        shard_file = f"{args.base_dir}/out/refinedweb_01_0/DataMan-annotation/shard_{str(i).zfill(8)}_processed.jsonl"
        with open(shard_file, "r") as f:
            for line in f:
                data = json.loads(line)
                # Extract overall_score, default to 0.0 if not present
                score = data.get("overall_score", 0.0) + data.get("creativity", 0.0)
                all_scores.append(score)

    all_scores = np.array(all_scores)
    print(f">> Loaded {len(all_scores)} scores")
    print(f">> Scores mean: {all_scores.mean()}")
    print(f">> Scores std: {all_scores.std()}")

    # count how many scores equal 0, 1, 2, 3, 4, 5
    unique, counts = np.unique(all_scores, return_counts=True)
    score_distribution = dict(zip(unique, counts))
    print(">> Score distribution:", score_distribution)

    # Select top indices based on overall_score
    indices = np.argpartition(-all_scores, selection_size)[:selection_size]
    return indices


def edu_select(dataset_size, selection_size, args):
    dataset = datasets.concatenate_datasets(
        [
            datasets.load_from_disk(f"{args.output_dir}/{i}")
            for i in range(args.shard_num)
        ]
    )
    metrics = np.array(dataset["prediction"]).reshape(-1)
    print(">> Metrics shape:", metrics.shape)
    print(">> Metrics mean:", metrics.mean())
    return np.argpartition(-metrics, selection_size)[:selection_size]


def random_select(dataset_size, selection_size, args):
    rng = np.random.default_rng()
    return rng.choice(dataset_size, size=(selection_size,), replace=False)


METHODS = {
    "random": random_select,
    "fineweb-edu": edu_select,
    "dataman": dataman_select,
}


def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    select_it = METHODS[args.method]
    ls = select_it(dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/project/flame/zichunyu")
    parser.add_argument("--method", type=str, default="dataman")
    parser.add_argument("--shard_num", type=float, default=1)
    parser.add_argument("--ratio", type=int, default=10)

    args = parser.parse_args()
    print(args)

    out_dir = Path(f"{args.base_dir}/out/refinedweb_01_0/DataMan-selection/processed_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "dataman":
        shard_names = [f"shard_{str(i).zfill(8)}_processed" for i in range(512)]
        file_dir = "s3://commoncrawl/contrib/datacomp/DCLM-refinedweb/global-shard_01_of_10/local-shard_0_of_10/{}.jsonl.zstd"
        dataset_size = 0
        shard_sizes = []
        for i in range(512):
            dataman_file = f"{args.base_dir}/out/refinedweb_01_0/DataMan-annotation/shard_{str(i).zfill(8)}_processed.jsonl"
            with open(dataman_file, "r") as f:
                count = sum(1 for _ in f)
                shard_sizes.append(count)
                dataset_size += count
    else:
        # Original logic for other methods
        data_dir = f"{args.base_dir}/data/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
        file_list = [
            os.path.abspath(os.path.join(data_dir, f))
            for f in os.listdir(data_dir)
            if not f.startswith(".")
        ]
        shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
        file_dir = "/project/flame/zichunyu/data/refinedweb_01_0/fasttext/fasttext_filter/processed_data/{}.jsonl.zstd"

        shard_sizes = []
        for shard_name in tqdm(shard_names):
            shard_file = file_dir.format(shard_name)
            count = sum(1 for _ in read_jsonl(shard_file))
            shard_sizes.append(count)
        dataset_size = sum(shard_sizes)

    print(f">> Total dataset size: {dataset_size}")
    # selection_size = dataset_size // args.ratio
    # selection_size = 8415673
    selection_size = 4069838 + 410909
    indices = get_indices(dataset_size, selection_size, args)
    selected_indices_set = set(indices)
    print(f">> Max selected index: {max(indices)}")

    global_offset = 0
    for shard_i, shard_name in tqdm(enumerate(shard_names)):
        in_file = file_dir.format(shard_name)
        out_file = out_dir / (shard_name + ".jsonl.zstd")
        out_data = []
        count = 0
        for line_idx, line in enumerate(read_jsonl(in_file)):
            count += 1
            global_idx = global_offset + line_idx
            if global_idx in selected_indices_set:
                out_data.append(line)
        assert (
            count == shard_sizes[shard_i]
        ), f"Shard {shard_name} size mismatch: {count} != {shard_sizes[shard_i]}"
        write_jsonl(out_data, str(out_file))
        global_offset += shard_sizes[shard_i]
