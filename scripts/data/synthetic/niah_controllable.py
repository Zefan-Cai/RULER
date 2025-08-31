# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Create a controllable dataset for needle in a haystack with precise length and depth control.

python niah_controllable.py \
    --save_dir=./ \
    --save_name=niah_controllable \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --test_max_length=32000 \
    --length_interval=1000 \
    --num_depth_positions=10 \
    --tokens_to_generate=128 \
    --num_samples=1 \
    --template="Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"
"""
import os
import re
import json
import uuid
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import wonderwords
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizer import select_tokenizer
from manifest_utils import write_manifest
from nltk.tokenize import sent_tokenize
import logging

import nltk
nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

from constants import TASKS

parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, default=1, help='number of samples per configuration')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, default="Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are", help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--keep_answer_prefix", action='store_true', help='keep answer_prefix in the input text instead of removing it.')

# Controllable Length and Depth Configurations
parser.add_argument("--test_max_length", type=int, required=True, help='maximum sequence length to test')
parser.add_argument("--length_interval", type=int, required=True, help='interval between test lengths')
parser.add_argument("--num_depth_positions", type=int, required=True, help='number of depth positions to test')
parser.add_argument("--start_length", type=int, default=None, help='starting length (default: length_interval)')

# Complexity Configurations
parser.add_argument("--num_needle_k", type=int, default=1)
parser.add_argument("--num_needle_v", type=int, default=1)
parser.add_argument("--num_needle_q", type=int, default=1)
parser.add_argument("--type_haystack", type=str, default='essay', help='[Options] noise, essay, needle.')
parser.add_argument("--type_needle_k", type=str, default='words', help='[Options] numbers, words, uuids.')
parser.add_argument("--type_needle_v", type=str, default='numbers', help='[Options] numbers, words, uuids.')
parser.add_argument("--model_template_token", type=int, default=0, help='used for nemo skills, minus num of model template token')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
args.num_needle_k = max(args.num_needle_k, args.num_needle_q)

# Set default start_length if not provided
if args.start_length is None:
    args.start_length = args.length_interval

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Define Needle/Haystack Format
needle = "One of the special magic {type_needle_v} for {key} is: {value}."
if args.type_haystack == 'essay':
    essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/PaulGrahamEssays.json")
    essay = json.load(open(essay))['text']
    haystack = re.sub(r'\s+', " ", essay).split(" ")
elif args.type_haystack == 'noise':
    haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
elif args.type_haystack == 'needle':
    haystack = needle
else:
    raise NotImplementedError(f'{args.type_haystack} is not implemented.')

# Words
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))


def generate_depth_positions(num_positions):
    """Generate evenly distributed depth positions (center points)"""
    if num_positions == 1:
        return [50.0]  # Single position at the middle
    
    # Calculate interval
    interval = 100.0 / num_positions
    
    # Generate center point positions
    positions = []
    for i in range(num_positions):
        center = interval * (i + 0.5)  # Center point of each interval
        positions.append(center)
    
    return positions


def generate_random_number(num_digits=7):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def generate_random_word():
    word = random.choice(words)
    return word

def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def generate_random(type_needle: str):
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    else:
        raise NotImplementedError(f'{args.type_needle} is not implemented.')


def generate_input_output_at_depth(num_haystack, depth_percent):
    """Generate input/output for a specific depth position"""
    keys, values, needles = [], [], []
    for _ in range(args.num_needle_k):
        keys.append(generate_random(args.type_needle_k))
        value = []
        for _ in range(args.num_needle_v):
            value.append(generate_random(args.type_needle_v))
            needles.append(needle.format(
                type_needle_v=args.type_needle_v,
                key=keys[-1],
                value=value[-1],
            ))
        values.append(value)

    random.shuffle(needles)  # Use the global random state instead of fixed seed

    # Context
    if args.type_haystack == 'essay':
        text = " ".join(haystack[:num_haystack])
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            # Repeat haystack as many times as needed and slice to num_haystack
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)  # Ceiling division
            text = " ".join((haystack * repeats)[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        
        # Insert needles at specified depth
        insertion_positions = []
        for _ in range(len(needles)):
            pos = int(len(document_sents) * (depth_percent / 100))
            insertion_positions.append(pos)
        insertion_positions = [0] + sorted(insertion_positions) + [len(document_sents)]
        
        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i-1 < len(needles):
                document_sents_list.append(needles[i-1])
        context = " ".join(document_sents_list)

    else:
        if args.type_haystack == 'noise':
            sentences = [haystack] * num_haystack
        elif args.type_haystack == 'needle':
            sentences = [haystack.format(
                type_needle_v=args.type_needle_v,
                key=generate_random(args.type_needle_k),
                value=generate_random(args.type_needle_v),
            ) for _ in range(num_haystack)]

        # Insert at specific depth
        insert_index = int(num_haystack * (depth_percent / 100))
        for needle_text in needles:
            sentences.insert(insert_index, needle_text)
        context = "\n".join(sentences)

    ## Query and Answer
    indices = random.sample(range(args.num_needle_k), args.num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]

    template = args.template
    type_needle_v = args.type_needle_v
    if args.num_needle_q * args.num_needle_v == 1:
        template = template.replace('Some', 'A')
        template = template.replace('are all', 'is')
        template = template.replace('are', 'is')
        template = template.replace('answers', 'answer')
        type_needle_v = type_needle_v[:-1] # remove "s"

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query,
    )

    # Calculate actual needle position in tokens
    needle_position = -1
    if len(answers) > 0:
        needle_index = input_text.find(answers[0])
        if needle_index != -1:
            needle_position = len(TOKENIZER.text_to_tokens(input_text[:needle_index]))

    return input_text, answers, needle_position


def estimate_haystack_size_for_length(target_length, incremental=500):
    """Estimate the number of haystack units needed for target token length"""
    # Generate a sample to estimate tokens per haystack unit
    sample_input, _, _ = generate_input_output_at_depth(incremental, 50)
    sample_tokens = len(TOKENIZER.text_to_tokens(sample_input))
    tokens_per_unit = sample_tokens / incremental
    
    # Estimate with some buffer
    estimated_units = int((target_length - args.tokens_to_generate) / tokens_per_unit * 0.9)
    return max(incremental, estimated_units)


def find_optimal_haystack_for_length(target_length, depth_percent):
    """Binary search to find optimal haystack size for target length"""
    if args.type_haystack == 'essay':
        incremental = 500
    elif args.type_haystack == 'noise':
        incremental = 25
    elif args.type_haystack == 'needle':
        incremental = 25
    
    if args.type_haystack != 'essay' and target_length < 4096:
        incremental = 5
    
    # Initial estimate
    lower_bound = incremental
    upper_bound = estimate_haystack_size_for_length(target_length, incremental)
    
    optimal_num_haystack = None
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        input_text, answer, _ = generate_input_output_at_depth(mid, depth_percent)
        total_tokens = len(TOKENIZER.text_to_tokens(input_text)) + args.tokens_to_generate
        
        if total_tokens <= target_length:
            # This size works, can we go larger?
            optimal_num_haystack = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1
    
    return optimal_num_haystack if optimal_num_haystack is not None else incremental


def generate_samples():
    """Generate samples for all length and depth combinations"""
    write_jsons = []
    
    # Generate test lengths
    test_lengths = list(range(args.start_length, args.test_max_length + 1, args.length_interval))
    
    # Generate depth positions
    depth_positions = generate_depth_positions(args.num_depth_positions)
    
    logger.info(f"Testing lengths: {test_lengths[:5]}...{test_lengths[-5:] if len(test_lengths) > 10 else test_lengths[5:]}")
    logger.info(f"Depth positions: {[f'{d:.1f}%' for d in depth_positions]}")
    logger.info(f"Total configurations: {len(test_lengths)} lengths × {len(depth_positions)} depths = {len(test_lengths) * len(depth_positions)}")
    
    # Progress bar
    total_samples = len(test_lengths) * len(depth_positions) * args.num_samples
    pbar = tqdm(total=total_samples, desc="Generating samples")
    
    sample_index = 0
    for target_length in test_lengths:
        # Adjust target length for model template tokens
        adjusted_length = target_length - args.model_template_token
        
        for depth_percent in depth_positions:
            # Find optimal haystack size for this length
            optimal_haystack = find_optimal_haystack_for_length(adjusted_length, depth_percent)
            
            for sample_num in range(args.num_samples):
                pbar.set_description(f"Length: {target_length}, Depth: {depth_percent:.1f}%, Sample: {sample_num+1}/{args.num_samples}")
                
                # Generate sample
                input_text, answers, needle_position = generate_input_output_at_depth(optimal_haystack, depth_percent)
                
                # Calculate actual length
                actual_length = len(TOKENIZER.text_to_tokens(input_text)) + args.tokens_to_generate
                
                if args.remove_newline_tab:
                    input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
                
                # Extract answer prefix
                answer_prefix_index = input_text.rfind(TASKS['niah']['answer_prefix'][:10])
                answer_prefix = input_text[answer_prefix_index:] if answer_prefix_index != -1 else ""
                if answer_prefix_index != -1 and not args.keep_answer_prefix:
                    input_text = input_text[:answer_prefix_index]
                
                # Create output record
                formatted_output = {
                    'index': sample_index,
                    "input": input_text,
                    "outputs": answers,
                    "length": actual_length,
                    'length_w_model_temp': actual_length + args.model_template_token,
                    'answer_prefix': answer_prefix,
                    'token_position_answer': needle_position,
                    'target_length': target_length,
                    'depth_percent': depth_percent,
                    'test_config': {
                        'max_length': args.test_max_length,
                        'interval': args.length_interval,
                        'num_positions': args.num_depth_positions,
                        'sample_num': sample_num
                    }
                }
                
                write_jsons.append(formatted_output)
                sample_index += 1
                pbar.update(1)
    
    pbar.close()
    
    # Print statistics
    logger.info(f"\nGeneration complete!")
    logger.info(f"Total samples generated: {len(write_jsons)}")
    logger.info(f"Average actual length: {np.mean([s['length'] for s in write_jsons]):.1f} tokens")
    logger.info(f"Length range: {min([s['length'] for s in write_jsons])} - {max([s['length'] for s in write_jsons])} tokens")
    
    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating controllable NIAH dataset...")
    logger.info(f"Max length: {args.test_max_length}, Interval: {args.length_interval}, Positions: {args.num_depth_positions}")
    
    write_jsons = generate_samples()
    
    write_manifest(save_file, write_jsons)
    logger.info(f"Dataset saved to: {save_file}")


if __name__ == "__main__":
    main()