# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RULER (What's the Real Context Size of Your Long-Context Language Models?) is a benchmark tool for evaluating long-context capabilities of language models through synthetic tasks. It generates configurable synthetic examples to test models across varying sequence lengths (4K to 131K tokens) and task complexities.

## Build and Development Commands

### Docker Setup
```bash
# Build Docker container
cd docker/
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t cphsieh/ruler:0.2.0 .

# Or use pre-built container
docker pull cphsieh/ruler:0.2.0
```

### Data Preparation
```bash
# Download required datasets
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
```

### Running Evaluations
```bash
# Basic evaluation command
cd scripts/
bash run.sh MODEL_NAME synthetic

# Configure these variables in run.sh:
GPUS="1"                    # Number of GPUs
ROOT_DIR="benchmark_root"   # Results storage directory
MODEL_DIR="../.."          # HuggingFace model directory
ENGINE_DIR="."             # TensorRT-LLM engine directory
```

### Testing Individual Tasks
```bash
# Generate synthetic data for specific task
python scripts/data/prepare.py \
    --save_dir output_dir \
    --benchmark synthetic \
    --task_name niah_single_1 \
    --tokenizer_path MODEL_PATH \
    --tokenizer_type MODEL_TYPE \
    --max_seq_length 32768 \
    --num_samples 100

# Run evaluation on generated data
python scripts/eval/evaluate.py \
    --data_dir output_dir \
    --benchmark synthetic
```

## High-Level Architecture

### Core Components

1. **Data Generation Framework** (`scripts/data/`):
   - `prepare.py`: Main data preparation script that orchestrates task generation
   - `synthetic/`: Task implementations (NIAH, QA, variable tracking, word extraction)
   - `template.py`: Model-specific chat templates (meta-chat, vicuna, llama3, etc.)
   - Generates synthetic examples with controlled complexity and sequence length

2. **Model Inference Layer** (`scripts/pred/`):
   - `call_api.py`: Unified API calling infrastructure
   - `model_wrappers.py`: Abstractions for different frameworks (HF, vLLM, OpenAI, Gemini)
   - `serve_vllm.py` / `serve_trt.py`: Model serving implementations
   - Supports multiple inference backends with consistent interface

3. **Evaluation Pipeline** (`scripts/eval/`):
   - `evaluate.py`: Automatic metric computation
   - Scoring functions: string_match_all, string_match_part
   - Aggregates results across sequence lengths and task complexities

4. **Configuration Management**:
   - `config_models.sh`: Model definitions with paths, templates, and tokenizers
   - `config_tasks.sh`: Task parameters (NUM_SAMPLES=500, temperature=0.0)
   - `synthetic.yaml`: Complete task specifications with complexity parameters
   - `run.sh`: Main pipeline orchestrator that coordinates all components

### Task Architecture

The benchmark implements 13 synthetic tasks across 4 categories:

1. **Retrieval (NIAH variants)**: Tests finding specific information in long contexts
   - Configurable needle types (words, numbers, UUIDs)
   - Different haystack types (noise, essays, needles)
   - Multi-key, multi-value, and multi-query variants

2. **Multi-hop Tracing**: Variable tracking through logical chains
   - Configurable number of chains and hops
   - Tests reasoning across document

3. **Aggregation**: Extract and count common/frequency words
   - Tests global context understanding

4. **Question Answering**: Long-context QA from SQuAD/HotpotQA
   - Tests comprehension with distractors

### Model Integration Flow

1. Model configuration in `config_models.sh` defines:
   - Model path and framework (vllm, hf, openai, gemini, trtllm)
   - Chat template type for proper formatting
   - Tokenizer configuration

2. Data preparation creates task-specific inputs:
   - Tokenizer-aware length control
   - Template application for model-specific formatting
   - Manifest generation for batch processing

3. Inference execution through appropriate backend:
   - vLLM: Tensor-parallel serving with optimizations
   - HuggingFace: Direct transformers with flash attention
   - API-based: OpenAI/Gemini cloud models
   - TensorRT-LLM: Optimized engine inference

4. Result evaluation computes:
   - Task-specific metrics (exact match, partial match)
   - Performance across sequence lengths
   - Aggregated benchmark scores

## Key Configuration Files

- `scripts/config_models.sh`: Model definitions and paths
- `scripts/config_tasks.sh`: Evaluation parameters
- `scripts/synthetic.yaml`: Task specifications with complexity parameters
- `scripts/data/template.py`: Chat templates for different model types
- `scripts/data/synthetic/constants.py`: Task templates and token limits

## Adding New Models

1. Add model configuration to `config_models.sh`:
```bash
case $MODEL_NAME in
    your_model_name)
        MODEL_PATH=${MODEL_DIR}/your_model_folder
        MODEL_TEMPLATE_TYPE="meta-chat"  # or appropriate template
        MODEL_FRAMEWORK="vllm"           # or hf, openai, gemini, trtllm
        TOKENIZER_TYPE="NousResearch/Meta-Llama-3-8B"  # for tokenizer
        TOKENIZER_PATH=${MODEL_PATH}
        ;;
esac
```

2. If needed, add custom chat template to `scripts/data/template.py`

3. Run evaluation: `bash run.sh your_model_name synthetic`

## Adding New Tasks

1. Implement task class in `scripts/data/synthetic/`:
   - Inherit from base task structure
   - Implement `build_description()` method
   - Define complexity parameters

2. Add task configuration to `scripts/synthetic.yaml`:
   - Define task parameters and complexity levels
   - Specify evaluation metrics

3. Update `scripts/config_tasks.sh` to include new task

4. Add evaluation logic to `scripts/eval/synthetic/constants.py` if needed

## Performance Considerations

- Default: 500 samples per task for statistical significance
- Greedy decoding (temperature=0.0) for reproducibility
- Sequence lengths: 4K, 8K, 16K, 32K, 65K, 131K tokens
- Flash attention recommended for long sequences
- Tensor parallelism for large models via vLLM

## Dependencies

Core requirements (from `docker/requirements.txt`):
- transformers==4.44.2
- vllm==0.5.4
- huggingface_hub==0.23.4
- openai, tiktoken (for API models)
- google-generativeai (for Gemini)
- accelerate, flash-attn (for efficiency)
- wonderwords (for synthetic data generation)