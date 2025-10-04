TOKENIZER_PATH="/mnt/localssd/zefan/TTT/flame_mini/fla-hub-local/transformer-1.3B-100B"  # Change this to your tokenizer path
TOKENIZER_TYPE="hf"
SAVE_DIR="/mnt/localssd/data"

python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_64k_1k_10pos_5samples_essay_words_numbers2 \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 64000 \
    --length_interval 1000 \
    --num_depth_positions 10 \
    --num_samples 5 \
    --tokens_to_generate 64 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --num_digits_needle_v=2 \
    --keep_answer_prefix