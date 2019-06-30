You can start the script following this intruction:
```shell
usage: train.py [-h] [--dataset DATASET] [--feature FEATURE]
                [--cap_json CAP_JSON] [--dic_json DIC_JSON]
                [--batch_size BATCH_SIZE] [--d_model D_MODEL] [--lr LR]
                [--max_len MAX_LEN] [--is_cuda IS_CUDA] [--task TASK]
                [--att ATT] [--src_posi SRC_POSI] [--restore RESTORE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET
  --feature FEATURE
  --cap_json CAP_JSON
  --dic_json DIC_JSON
  --batch_size BATCH_SIZE
                        batch_size
  --d_model D_MODEL     batch_size
  --lr LR               learning rate
  --max_len MAX_LEN     max length
  --is_cuda IS_CUDA     max length
  --task TASK           task name for ckpt
  --att ATT             self-att for encoder
  --src_posi SRC_POSI   add postional encoding on src matrix
  --restore RESTORE
```