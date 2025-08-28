# 基于 niah.py 的可控参数脚本开发计划

## 1. 项目背景与目标

### 背景
当前的 `niah.py` 脚本使用固定的 40 个深度位置（0-100%）和自动二分搜索来确定最佳 haystack 大小。这种方式虽然自动化程度高，但缺乏对特定长度和位置的精确控制。

### 目标
开发一个新的脚本 `niah_controlled.py`，提供以下可控参数：
- **可控长度**：如 32000 或 128000 tokens
- **可控长度间隔**：如每 1000 或 2000 tokens 生成一个测试点
- **可控插入位置数量**：如在每个长度上测试 5 或 10 个不同的 needle 位置

## 2. 核心功能设计

### 2.1 参数控制系统

```python
# 新增的核心控制参数
--target_length: int        # 目标最大长度（如 32000, 128000）
--length_interval: int      # 长度间隔（如 1000, 2000）  
--num_positions: int        # 每个长度的插入位置数量（如 5, 10）
--position_strategy: str    # 位置策略：'uniform' 或 'random'
--start_length: int         # 起始长度（默认等于 length_interval）
```

### 2.2 生成策略

#### 长度序列生成
```python
lengths = [start_length + i * length_interval 
           for i in range((target_length - start_length) // length_interval + 1)]
# 例如：target_length=128000, interval=1000, start_length=1000
# 生成：[1000, 2000, 3000, ..., 128000]
```

#### 位置列表生成
```python
# Uniform 策略：均匀分布
positions = [i * 100 / num_positions for i in range(1, num_positions + 1)]
# 例如：num_positions=10 → [10%, 20%, 30%, ..., 100%]

# Random 策略：随机选择
positions = sorted(random.sample(range(5, 100, 5), num_positions))
```

## 3. 技术实现方案

### 3.1 文件结构

```
scripts/data/synthetic/
├── niah.py                 # 原始文件（保持不变）
├── niah_controlled.py      # 新的可控参数版本
└── controlled_output/      # 输出目录
    ├── metadata.json       # 元数据文件
    ├── length_1000/
    │   ├── position_10.jsonl
    │   ├── position_20.jsonl
    │   └── ...
    ├── length_2000/
    │   └── ...
    └── summary.json        # 汇总统计信息
```

### 3.2 核心函数设计

#### 函数 1：长度序列生成器
```python
def generate_length_sequence(target_length, interval, start_length=None):
    """
    生成测试长度序列
    
    Args:
        target_length: 目标最大长度
        interval: 长度间隔
        start_length: 起始长度（默认为 interval）
    
    Returns:
        List[int]: 长度序列列表
    """
    if start_length is None:
        start_length = interval
    
    lengths = []
    current = start_length
    while current <= target_length:
        lengths.append(current)
        current += interval
    
    return lengths
```

#### 函数 2：位置列表生成器
```python
def generate_position_list(num_positions, strategy='uniform'):
    """
    生成 needle 插入位置列表
    
    Args:
        num_positions: 位置数量
        strategy: 'uniform' 或 'random'
    
    Returns:
        List[float]: 位置百分比列表（0-100）
    """
    if strategy == 'uniform':
        # 均匀分布：10%, 20%, ..., 100%
        return [i * 100.0 / num_positions 
                for i in range(1, num_positions + 1)]
    elif strategy == 'random':
        # 随机选择不重复的位置
        candidates = list(range(5, 100, 5))  # 5% 的倍数
        return sorted(random.sample(candidates, 
                     min(num_positions, len(candidates))))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

#### 函数 3：精确长度和位置的样本生成器
```python
def generate_sample_at_position(target_tokens, position_percent, 
                                needle_content, haystack_type, tokenizer):
    """
    在指定的 token 长度和位置生成样本
    
    Args:
        target_tokens: 目标 token 数量
        position_percent: needle 插入位置（0-100）
        needle_content: needle 内容
        haystack_type: haystack 类型
        tokenizer: tokenizer 实例
    
    Returns:
        dict: 包含 input、output、metadata 的样本
    """
    # 1. 估算需要的 haystack 单元数量
    # 2. 生成 haystack 内容
    # 3. 在精确位置插入 needle
    # 4. 验证总 token 数
    # 5. 返回格式化的样本
```

### 3.3 主要流程

```python
def main():
    # 1. 解析参数
    args = parse_arguments()
    
    # 2. 初始化 tokenizer
    tokenizer = select_tokenizer(args.tokenizer_type, args.tokenizer_path)
    
    # 3. 生成长度和位置序列
    lengths = generate_length_sequence(
        args.target_length, 
        args.length_interval,
        args.start_length
    )
    positions = generate_position_list(
        args.num_positions,
        args.position_strategy
    )
    
    # 4. 创建输出目录结构
    create_output_structure(args.save_dir, lengths, positions)
    
    # 5. 生成数据集
    metadata = {
        'target_length': args.target_length,
        'length_interval': args.length_interval,
        'num_positions': args.num_positions,
        'position_strategy': args.position_strategy,
        'lengths': lengths,
        'positions': positions,
        'samples_per_config': args.num_samples,
        'total_samples': len(lengths) * len(positions) * args.num_samples
    }
    
    # 6. 批量生成样本
    for length in tqdm(lengths, desc="Lengths"):
        for position in tqdm(positions, desc="Positions", leave=False):
            samples = []
            for i in range(args.num_samples):
                sample = generate_sample_at_position(
                    target_tokens=length,
                    position_percent=position,
                    needle_content=generate_needle_content(args),
                    haystack_type=args.type_haystack,
                    tokenizer=tokenizer
                )
                samples.append(sample)
            
            # 保存到对应的文件
            save_samples(samples, args.save_dir, length, position)
    
    # 7. 保存元数据
    save_metadata(metadata, args.save_dir)
```

## 4. 实现步骤详解

### 第 1 步：创建基础框架
1. 复制 `niah.py` 为 `niah_controlled.py`
2. 保留必要的导入和工具函数
3. 移除二分搜索相关代码

### 第 2 步：修改参数系统
```python
# 新增参数
parser.add_argument("--target_length", type=int, required=True, 
                   help='Target maximum sequence length in tokens')
parser.add_argument("--length_interval", type=int, required=True,
                   help='Interval between test lengths')
parser.add_argument("--num_positions", type=int, required=True,
                   help='Number of needle positions per length')
parser.add_argument("--position_strategy", type=str, default='uniform',
                   choices=['uniform', 'random'],
                   help='Strategy for position selection')
parser.add_argument("--start_length", type=int, default=None,
                   help='Starting length (default: length_interval)')

# 保留的原始参数
parser.add_argument("--type_haystack", ...)
parser.add_argument("--type_needle_k", ...)
parser.add_argument("--type_needle_v", ...)
# ...
```

### 第 3 步：实现核心生成逻辑

#### 精确控制 token 数量
```python
def calculate_haystack_size(target_tokens, needle_tokens, 
                           template_tokens, tokenizer, haystack_type):
    """
    计算达到目标 token 数所需的 haystack 大小
    使用迭代方法逐步调整
    """
    if haystack_type == 'essay':
        # 对于 essay，使用词数控制
        return estimate_essay_words(target_tokens, needle_tokens, 
                                  template_tokens, tokenizer)
    else:
        # 对于 noise 和 needle，使用句子数控制
        return estimate_sentence_count(target_tokens, needle_tokens,
                                      template_tokens, tokenizer)
```

#### 精确控制插入位置
```python
def insert_needle_at_position(haystack_content, needle_content, 
                             position_percent):
    """
    在 haystack 的指定百分比位置插入 needle
    """
    if isinstance(haystack_content, str):
        sentences = sent_tokenize(haystack_content)
    else:
        sentences = haystack_content
    
    # 计算插入位置索引
    insert_index = int(len(sentences) * position_percent / 100)
    insert_index = min(max(insert_index, 0), len(sentences))
    
    # 插入 needle
    result = sentences[:insert_index] + [needle_content] + sentences[insert_index:]
    
    return ' '.join(result) if isinstance(haystack_content, str) else result
```

### 第 4 步：输出格式设计

#### JSONL 格式
每个样本包含以下字段：
```json
{
    "index": 0,
    "input": "...",
    "outputs": ["answer1", "answer2"],
    "length": 32000,
    "actual_length": 31998,
    "target_position": 50.0,
    "actual_position": 49.8,
    "needle_token_position": 15999,
    "answer_prefix": "...",
    "metadata": {
        "haystack_type": "essay",
        "needle_type_k": "words",
        "needle_type_v": "numbers",
        "generation_time": "2024-01-01T12:00:00"
    }
}
```

#### 元数据格式
```json
{
    "configuration": {
        "target_length": 128000,
        "length_interval": 1000,
        "num_positions": 10,
        "position_strategy": "uniform",
        "start_length": 1000
    },
    "generation_stats": {
        "total_samples": 12800,
        "lengths_tested": [1000, 2000, ...],
        "positions_tested": [10.0, 20.0, ...],
        "generation_time": "2024-01-01T12:00:00",
        "tokenizer": "meta-llama/Llama-3-8B"
    },
    "sample_distribution": {
        "samples_per_length": 100,
        "samples_per_position": 10,
        "total_configurations": 1280
    }
}
```

## 5. 测试计划

### 5.1 单元测试
```python
def test_length_sequence_generation():
    """测试长度序列生成"""
    lengths = generate_length_sequence(10000, 2000, 2000)
    assert lengths == [2000, 4000, 6000, 8000, 10000]

def test_position_list_generation():
    """测试位置列表生成"""
    positions = generate_position_list(5, 'uniform')
    assert positions == [20.0, 40.0, 60.0, 80.0, 100.0]

def test_needle_insertion_accuracy():
    """测试 needle 插入位置准确性"""
    # 验证插入位置误差在 ±1% 以内
    pass
```

### 5.2 集成测试
```bash
# 小规模测试
python niah_controlled.py \
    --save_dir ./test_output \
    --tokenizer_path /path/to/tokenizer \
    --target_length 8000 \
    --length_interval 2000 \
    --num_positions 5 \
    --num_samples 2

# 验证输出结构
ls -la test_output/
cat test_output/metadata.json
```

### 5.3 性能基准测试
- 测试 128K 长度生成时间
- 监控内存使用峰值
- 验证 token 计数准确性

## 6. 使用示例

### 基础示例
```bash
# 生成 32K 长度的测试数据，每 2000 tokens 一个测试点
python scripts/data/synthetic/niah_controlled.py \
    --save_dir ./controlled_niah_32k \
    --save_name niah_controlled \
    --tokenizer_path /path/to/llama-3-tokenizer \
    --tokenizer_type hf \
    --target_length 32000 \
    --length_interval 2000 \
    --num_positions 5 \
    --position_strategy uniform \
    --num_samples 10 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers
```

### 高级示例 - 128K 完整测试
```bash
# 生成 128K 长度的完整测试集
python scripts/data/synthetic/niah_controlled.py \
    --save_dir ./controlled_niah_128k \
    --save_name niah_controlled_full \
    --tokenizer_path /path/to/llama-3-tokenizer \
    --tokenizer_type hf \
    --target_length 128000 \
    --length_interval 1000 \
    --start_length 1000 \
    --num_positions 10 \
    --position_strategy uniform \
    --num_samples 100 \
    --type_haystack essay \
    --type_needle_k words \
    --type_needle_v numbers \
    --num_needle_k 1 \
    --num_needle_v 1 \
    --num_needle_q 1 \
    --tokens_to_generate 128
```

### 自定义位置测试
```bash
# 使用随机位置策略
python scripts/data/synthetic/niah_controlled.py \
    --save_dir ./random_position_test \
    --save_name niah_random \
    --tokenizer_path /path/to/tokenizer \
    --tokenizer_type hf \
    --target_length 16000 \
    --length_interval 4000 \
    --num_positions 8 \
    --position_strategy random \
    --num_samples 50
```

## 7. 预期输出结构

```
controlled_niah_128k/
├── metadata.json                  # 全局元数据
├── summary.json                   # 汇总统计
├── length_1000/
│   ├── position_10.0.jsonl       # 1000 tokens, 10% 位置
│   ├── position_20.0.jsonl       # 1000 tokens, 20% 位置
│   └── ...
├── length_2000/
│   ├── position_10.0.jsonl
│   └── ...
└── length_128000/
    ├── position_10.0.jsonl
    └── position_100.0.jsonl
```

## 8. 性能优化建议

### 8.1 并行处理
```python
from multiprocessing import Pool

def generate_samples_parallel(configs):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(generate_single_config, configs)
    return results
```

### 8.2 缓存机制
```python
# 缓存 tokenizer 结果
@lru_cache(maxsize=10000)
def cached_tokenize(text, tokenizer_id):
    return tokenizer.text_to_tokens(text)
```

### 8.3 增量生成
```python
def resume_generation(save_dir):
    """从上次中断的地方继续生成"""
    existing_files = check_existing_files(save_dir)
    remaining_configs = get_remaining_configs(existing_files)
    continue_generation(remaining_configs)
```

## 9. 评估脚本集成

### 9.1 配套评估脚本
```python
# evaluate_controlled.py
def evaluate_controlled_results(result_dir, model_outputs):
    """
    评估模型在不同长度和位置的表现
    生成热力图显示性能分布
    """
    results = []
    for length_dir in result_dir.glob("length_*"):
        length = int(length_dir.name.split("_")[1])
        for position_file in length_dir.glob("position_*.jsonl"):
            position = float(position_file.stem.split("_")[1])
            accuracy = evaluate_file(position_file, model_outputs)
            results.append({
                'length': length,
                'position': position,
                'accuracy': accuracy
            })
    
    return create_heatmap(results)
```

### 9.2 可视化工具
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_performance(results):
    """
    创建性能热力图
    X轴：长度
    Y轴：位置
    颜色：准确率
    """
    pivot_table = results.pivot(
        index='position', 
        columns='length', 
        values='accuracy'
    )
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', 
                cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('NIAH Performance Heatmap')
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Needle Position (%)')
    plt.savefig('niah_performance_heatmap.png')
```

## 10. 常见问题处理

### Q1: Token 数量不精确
**问题**：实际生成的 token 数与目标不完全匹配
**解决**：
- 使用迭代调整方法
- 设置容差范围（如 ±1%）
- 记录实际 token 数供分析

### Q2: 内存溢出
**问题**：处理 128K 长度时内存不足
**解决**：
- 分批处理
- 使用生成器模式
- 及时释放不需要的对象

### Q3: 位置偏移
**问题**：实际插入位置与目标位置有偏差
**解决**：
- 使用句子级别的精确定位
- 记录实际位置供后续分析
- 提供位置校准功能

## 11. 未来扩展

1. **多 needle 支持**：在同一样本中插入多个 needle
2. **动态难度调整**：根据模型表现自动调整测试难度
3. **实时评估**：生成时同步进行模型评估
4. **分布式生成**：支持多机并行生成大规模数据集
5. **自适应采样**：在性能边界区域增加采样密度

## 12. 总结

本计划提供了一个完整的可控 NIAH 数据生成方案，主要特点：
- **精确控制**：长度、间隔、位置完全可控
- **灵活配置**：支持多种生成策略
- **结构化输出**：便于后续分析和评估
- **可扩展性**：易于添加新功能和优化

通过这个新脚本，研究人员可以更精确地评估模型在不同长度和位置条件下的表现，为长文本理解能力的改进提供数据支持。