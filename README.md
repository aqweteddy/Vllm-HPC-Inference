## 支援 slurm HPC 的 VLLM infernce 腳本


## Environment

* 你必須先準備好 vllm singularity image
* 修改 `bash/batch_run.sh` 中的 `SIF_PATH`


## Usage

```bash
bash bash/batch_run.bash <input_jsonl_file> <number of nodes> <output folder>   <py scropt>
bash bash/batch_run.bash <input_jsonl_file> 10 <output folder>  scripts/translation/translate_codefeedback.py
```

* input_jsonl_file: 輸入的 jsonl 檔案
* number of nodes: 使用的節點數量
* output folder: 輸出的資料夾
* py script: 要執行的 python script
