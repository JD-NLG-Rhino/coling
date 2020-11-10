copy=True
coverage=False
dev_batch_size=20
exclusive_words_file="data/exclusive_words_${5}"

CUDA_VISIBLE_DEVICES=$1  python -u nmt.py \
   --mode decode \
   --cuda \
   --copy ${copy} \
   --coverage ${coverage} \
   --dev-batch-size ${dev_batch_size} \
   --exclusive-words-file ${exclusive_words_file} \
    $2 \
    $3 \
    $4
