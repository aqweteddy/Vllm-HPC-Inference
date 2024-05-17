#!/bin/bash

# 檢查命令行參數
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file.jsonl> <num_parts> <process_command> <python_script>"
    exit 1
fi

# 獲取命令行參數
input_file="$1"
num_parts="$2"
output_dir="$3"
SIF_PATH=<SIF_PATH>
# 分割文件
rm -rf ./.cache/sbatch
mkdir -p ./.cache/sbatch
split -d -n l/$num_parts --numeric-suffixes --additional-suffix=".jsonl" $input_file  ./.cache/sbatch/part_


# 處理每個部分

SBATCH_ARGS=(
    --job-name=vllm
    --account=GOV112004
    --partition=gpNCHC_LLM
    --nodes=1
    --gpus-per-node=8
    --ntasks-per-node=1
    --cpus-per-task=32
)

PY_COMMAND=(
    python
    $4
)

SINGULARITY=(
    singularity run
    --nv
    --pwd $PWD
    "$SIF_PATH"
)

mkdir logs
mkdir $output_dir
for part in ./.cache/sbatch/part_*; do
    echo "Processing $part..."
    # get last after / in $part
    file="${part##*/}"
    py_file="${4##*/}"
    CMD="${SINGULARITY[@]} ${PY_COMMAND[@]} $part $output_dir/$file"
    echo $CMD
    sbatch ${SBATCH_ARGS[@]} --output logs/$py_file-$file.log --wrap "$CMD"
    # eval "$process_command" < "$part"
    # break
done

# 清理臨時文件
