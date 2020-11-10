# RecAdam

## Introduction

We provide **RecAdam** (Recall Adam) optimizer to facilitate fine-tuning deep pretrained language models (e.g., BERT, ALBERT) with less forgetting.

For a detailed description and experimental results, please refer to our paper: [Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting](https://www.aclweb.org/anthology/2020.emnlp-main.634/) (Accepted by EMNLP 2020).

## Environment

```bash
python >= 3.6
pytorch >= 1.0.0
transformers >= 2.5.1
```
## Files

- `RecAdam.py`: this file includes the RecAdam optimizer implementation, 
which is modified from AdamW optimizer implementation [`optimization.py`](https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py) by [Huggingface Transformers](https://github.com/huggingface/transformers).

- `run_glue_with_RecAdam.py`: this file is an example to run GLUE tasks with RecAdam optimizer,
and is modified from the GLUE example [`run_glue.py`](https://github.com/huggingface/transformers/blob/c44a17db1b87e31ad4c232e48d19a2700e8b690d/examples/run_glue.py) by [Huggingface Transformers](https://github.com/huggingface/transformers). 

## Run GLUE tasks

GLUE tasks can be download from
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpacked to some directory `$GLUE_DIR`.

### With ALBERT-xxlarge model

For ALBERT-xxlarge, we use the same hyperparameters following [ALBERT paper](https://arxiv.org/pdf/1909.11942.pdf),
except for the maximum sequence length, which we set to 128 rather than 512.

As for the hyperparameters of RecAdam, 
we choose the sigmoid annealing function,
set the coefficient of the quadratic penalty to 5,000, 
select the best k in \{0.05, 0.1, 0.2, 0.5, 1\},
select the best t_0 in \{100, 250, 500\} for small tasks and \{250, 500, 1,000} for large tasks.

Here is an example script to get started:
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
  --albert_dropout 0.0 \
  --optimizer RecAdam \
  --recadam_anneal_fun sigmoid \
  --recadam_anneal_t0 1000 \
  --recadam_anneal_k 0.1 \
  --recadam_pretrain_cof 5000.0 \
  --logging_Euclid_dist 
```

### With BERT-base model

For BERT-base, we use the same hyperparameters following [BERT paper](https://arxiv.org/pdf/1810.04805.pdf).
We set the learning rate to 2e-5, and find that the model has not converged on each GLUE task after 3 epochs fine-tuning.
To make sure the convergence of vanilla fine-tuning, we increase the training step for each task 
(61,360 on MNLI, 56,855 on QQP, 33,890 on QNLI, 21,050 on SST, 13,400 on CoLA, 9,000 on STS, 11,500 on MRPC, 7,800 on RTE),
 and achieve better baseline scores on the dev set of GLUE benchmark.  

As for the hyperparameters of RecAdam, 
we choose the sigmoid annealing function,
set the coefficient of the quadratic penalty to 5,000, 
select the best k and t_0 in \{0.05, 0.1, 0.2, 0.5, 1\} and \{250, 500, 1,000\} respectively. 

Here is an example script to get started:
```bash
export GLUE_DIR=/path/to/glue
export TASK_NAME=STS-B

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
  --recadam_anneal_k 0.1 \
  --recadam_pretrain_cof 5000.0 \
  --logging_Euclid_dist 
```

### Citation
If you find RecAdam useful, please cite [our paper](https://www.aclweb.org/anthology/2020.emnlp-main.634/):
```bibtex
@inproceedings{recadam,
    title = "Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting",
    author = "Chen, Sanyuan  and  Hou, Yutai  and  Cui, Yiming  and  Che, Wanxiang  and  Liu, Ting  and  Yu, Xiangzhan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.634",
    pages = "7870--7881",
}
```
