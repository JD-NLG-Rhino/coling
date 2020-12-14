copy=True
coverage=False
exclusive_words_file="data/exclusive_words_${5}"
test_kb_file="data/test_kb_acl_10"
dev_batch_size=20

CUDA_VISIBLE_DEVICES=$1  python -u nmt.py \
   --mode decode \
   --cuda \
   --copy ${copy} \
   --coverage ${coverage} \
   --exclusive-words-file ${exclusive_words_file} \
   --test-kb-table-file ${test_kb_file} \
    $2 \
    $3 \
    $4
