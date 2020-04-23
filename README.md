# RecAdam

## Run on GLUE with ALBERT-xxlarge

```bash
export GLUE_DIR=/path/to/glue
export TASK_NAME=CoLA

python run_glue_with_RecAdam.py \
  --model_type albert \
  --model_name_or_path /path/to/model \
  --log_path /path/to/log \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 1e-5 \
  --warmup_steps 320 \
  --max_steps 5336 \
  --output_dir /path/to/output/$TASK_NAME/ \
  --evaluate_during_training \
  --train_logging_steps 25 \
  --eval_logging_steps 100 \
  --optimizer RecAdam \
  --recadam_anneal_fun sigmoid \
  --recadam_anneal_t0 1000 \
  --recadam_anneal_k 0.2 \
  --logging_Euclid_dist \
  --albert_dropout 0.0 
```

## Run on GLUE with BERT

```bash
export GLUE_DIR=/path/to/glue
export TASK_NAME=STS

python run_glue_with_RecAdam.py \
  --model_type bert \
  --model_name_or_path /path/to/model \
  --log_path /path/to/log \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --max_steps 9000 \
  --output_dir /path/to/output/$TASK_NAME/ \
  --evaluate_during_training \
  --train_logging_steps 50 \
  --eval_logging_steps 180 \
  --optimizer RecAdam \
  --recadam_anneal_fun sigmoid \
  --recadam_anneal_t0 1000 \
  --recadam_anneal_k 1 \
  --logging_Euclid_dist 
```