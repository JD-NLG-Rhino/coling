#!/bin/sh
train_true=$1
gpu_id=$2
seed=$3
aspect_active=$4
aspect_word_active=$4
load_model=$5
num_trial=$6
best_r2=$7
best_ppl=$8
tri_grams_rank_active=1
word_repeat_active=1
tri_grams_limit_active=0
label_smoothing=0
dropout=0.2
#copy=False
copy=True
coverage=False
batch_size=16
patience=7
max_num_trial=10

if [ ${train_true} = 1 ]
then
    logfile=train_log_gpu${gpu_id}
    model_dir=models_gpu${gpu_id}
    mkdir -p ${model_dir}
    echo Start to train with TRUE data ... >> ${logfile}
    echo Log to ${logfile} >> ${logfile}
    echo Save model to ${model_dir} >> ${logfile}
    vocab="data/vocab.json"
    train_src="data/fushi_train_source_biaodian_final"
    train_tgt="data/fushi_train_target_biaodian_final"
    dev="data/my_dev_corpus_aspect"
    aspect_active=${aspect_active}
    aspect_word_active=${aspect_word_active}
    valid_niter=4000
else
    logfile=toy_train_log
    model_dir=toy_models
    mkdir -p ${model_dir}
    echo Start to train with TOY data ...
    echo logging to ${logfile}
    echo Save to ${model_dir}
    vocab="data/vocab.json"
    train_src="data/jiadian_train_source_100"
    train_tgt="data/jiadian_train_target_100"
    dev="data/my_dev_corpus_aspect_10"
    aspect_active=${aspect_active}
    aspect_word_active=${aspect_word_active}
    valid_niter=10
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python nmt.py \
    --gpu_id ${gpu_id} \
    --mode train \
    --cuda \
    --load-model ${load_model} \
    --num-trial=${num_trial} \
    --best-r2=${best_r2} \
    --best-ppl=${best_ppl} \
    --seed ${seed} \
    --copy ${copy} \
    --coverage ${coverage} \
    --aspect-active=${aspect_active} \
    --aspect-word-active=${aspect_word_active} \
    --word-repeat-active=${word_repeat_active} \
    --tri-grams-limit-active=${tri_grams_limit_active} \
    --tri-grams-rank-active=${tri_grams_rank_active} \
    --patience ${patience} \
    --max-num-trial ${max_num_trial} \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev ${dev} \
    --input-feed \
    --valid-niter ${valid_niter} \
    --batch-size ${batch_size} \
    --hidden-size 512 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --label-smoothing ${label_smoothing} \
    --dropout ${dropout} \
    --clip-grad 2.0 \
    --save-to ${model_dir}/model.bin \
    --lr-decay 0.5 \
    2>&1 | tee -a ${logfile}
