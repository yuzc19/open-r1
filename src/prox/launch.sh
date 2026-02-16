#!/bin/bash
#SBATCH --job-name=prox_doc_refining_xs
#SBATCH --output=<expected_output_file>
#SBATCH --partition=<your_partition>
#SBATCH --error=<expected_error_file>
#SBATCH --time=50:00:00
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

# export NNODE=8
# export NGPU=8
# export TOTAL_SPLIT=$((NNODE*NGPU))

# cmd="
# for i in \$(seq 0 \$((NGPU-1))); do
#     TOTAL_SPLIT=$TOTAL_SPLIT \\
#     NODE_GPUS=$NGPU \\
#     NODE_RANK=\$SLURM_NODEID \\
#     CUDA_VISIBLE_DEVICES=\$i \\
#     python -m data_gen.tasks.apply_doc_refining \\
#         --data_format jsonl.gz \\
#         --limit -1 \\
#         --model_path gair-prox/web-doc-refining-lm \\
#         --config_path data_gen/configs/apply_doc_refining.yaml \\
#         > ./logging/apply_doc_refining_\${SLURM_NODEID}_\${i}.log 2>&1 &
# done
# wait
# "

# echo "Executing command:"
# echo "$cmd"

# srun bash -c "$cmd"

# # ****************************************************
# # scripts for single node: (debug)
# # ****************************************************

export VLLM_USE_V1=0
export NNODE=2
export NGPU=8
# total split (int) = nnode * ngpu, write in shell expression
export TOTAL_SPLIT=$((NNODE*NGPU))
export SLURM_NODEID=1
for i in $(seq 0 $((NGPU-1))); do
  TOTAL_SPLIT=$TOTAL_SPLIT NODE_GPUS=$NGPU NODE_RANK=$SLURM_NODEID CUDA_VISIBLE_DEVICES=$i \
  PYTHONPATH=$PWD/src/prox python -m src.prox.apply_chunk_refining \
    --data_format jsonl.zstd \
    --limit -1 \
    --model_path gair-prox/web-chunk-refining-lm \
    --config_path src/prox/apply_chunk_refining.yaml \
    > /tmp/logs/apply_chunk_refining_${SLURM_NODEID}_${i}.log 2>&1 &
done