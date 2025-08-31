#!/bin/bash

# Example usage script for niah_controllable.py

# Set common parameters
TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B"  # Change this to your tokenizer path
TOKENIZER_TYPE="hf"
SAVE_DIR="./controllable_output"

# Example 1: Test 32K context with 2K intervals and 5 depth positions
# This will test at lengths: 2000, 4000, 6000, ..., 32000
# And at depths: 10%, 30%, 50%, 70%, 90%
echo "Example 1: Testing 32K context with 2K intervals and 5 positions"
python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_32k_2k_5pos \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 32000 \
    --length_interval 2000 \
    --num_depth_positions 5 \
    --num_samples 2 \
    --tokens_to_generate 128 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v words

# Example 2: Test 128K context with 1K intervals and 10 depth positions
# This will test at lengths: 1000, 2000, 3000, ..., 128000
# And at depths: 5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95%
echo "Example 2: Testing 128K context with 1K intervals and 10 positions"
python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_128k_1k_10pos \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 128000 \
    --length_interval 1000 \
    --num_depth_positions 10 \
    --num_samples 1 \
    --tokens_to_generate 128 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --template "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"

# Example 3: Fine-grained test for 16K context with 500 token intervals and 20 positions
# This will create a very detailed heatmap
echo "Example 3: Fine-grained 16K test with 500 token intervals and 20 positions"
python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_16k_500_20pos \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 16000 \
    --length_interval 500 \
    --num_depth_positions 20 \
    --num_samples 1 \
    --tokens_to_generate 128 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --template "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"

# Example 4: Multi-needle test with controllable positions
echo "Example 4: Multi-needle test with 4 keys and 4 values"
python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_multi_32k_2k_5pos \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 32000 \
    --length_interval 2000 \
    --num_depth_positions 5 \
    --num_samples 1 \
    --tokens_to_generate 256 \
    --num_needle_k 4 \
    --num_needle_v 4 \
    --num_needle_q 2 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --template "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"

# Example 5: Noise haystack test
echo "Example 5: Noise haystack test"
python scripts/data/synthetic/niah_controllable.py \
    --save_dir ${SAVE_DIR} \
    --save_name niah_noise_8k_500_10pos \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --test_max_length 8000 \
    --length_interval 500 \
    --num_depth_positions 10 \
    --num_samples 1 \
    --tokens_to_generate 128 \
    --type_haystack noise \
    --type_needle_k words \
    --type_needle_v numbers \
    --template "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"

echo "All examples completed!"