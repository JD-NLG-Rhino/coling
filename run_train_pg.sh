#!/bin/sh
train_true=$1
gpu_id=$2
seed=$3
label_smoothing=0
dropout=0.2
copy=True
coverage=False
batch_size=16
patience=7
max_num_trial=10
dev_batch_size=200

if [ ${train_true} = 1 ]
then
    logfile=train_log_gpu${gpu_id}
    model_dir=models_gpu${gpu_id}
    mkdir -p ${model_dir}
    echo Start to train with TRUE data ... > ${logfile}
    echo Log to ${logfile} >> ${logfile}
    echo Save model to ${model_dir} >> ${logfile}
    vocab="data/vocab.json"
    train_src="data/train_source_acl"
    train_tgt="data/train_target_acl"
    dev="data/dev_courpus_acl"
    valid_niter=4000
    exclusive_words="data/exclusive_words_${4}"
else
    logfile=toy_train_log
    model_dir=toy_models
    mkdir -p ${model_dir}
    echo Start to train with TOY data ...
    echo logging to ${logfile}
    echo Save to ${model_dir}
    vocab="data/vocab.json"
    train_src="data/train_source_acl_500"
    train_tgt="data/train_target_acl_500"
    dev="data/dev_courpus_acl_50"
    valid_niter=100
    exclusive_words="data/exclusive_words_${4}"
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python -u nmt.py \
    --gpu_id ${gpu_id} \
    --mode train \
    --cuda \
    --seed ${seed} \
    --copy ${copy} \
    --coverage ${coverage} \
    --patience ${patience} \
    --max-num-trial ${max_num_trial} \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-batch-size ${dev_batch_size} \
    --dev ${dev} \
    --exclusive-words-file ${exclusive_words} \
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
