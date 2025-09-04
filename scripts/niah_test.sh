TOKENIZER_PATH="/mnt/localssd/TTT/flame_mini/fla-hub-local/transformer-1.3B-100B"  # Change this to your tokenizer path
TOKENIZER_TYPE="hf"
SAVE_DIR="/mnt/localssd/RULER/controllable_output"

python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_128k_1k_5pos_1samples \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 128000 \
    --length_interval 1000 \
    --num_depth_positions 5 \
    --num_samples 1 \
    --tokens_to_generate 64 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --keep_answer_prefix