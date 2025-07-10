begin=$1
end=$2
gpu_index=0
for ((s=begin; s<=end; s++)); do
    echo $s
    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=$gpu_index python src/infer/run_infer.py $s \
    > logs/log_job_s${s}_gpu${gpu_index}.out 2>&1 &
    ((gpu_index=(gpu_index+1)%8))
done

# cat transformed_texts_polish_*.jsonl > transformed_texts_polish.jsonl