## 计划：创建可定制的 NIAH 数据生成脚本

**目标**: 创建一个新的 Python 脚本 `scripts/data/synthetic/configurable_niah.py`，该脚本可以根据用户指定的参数生成“大海捞针”测试数据。

**可配置参数**:
1.  `--max-length`: 上下文的最大长度 (例如: 128000)。
2.  `--length-interval`: 长度递增的步长 (例如: 1000)。
3.  `--needle-positions`: 在每个特定长度的上下文中，需要插入“needle”的位置数量 (例如: 10)。

---

### **执行步骤**

#### **第一步：分析现有脚本 `niah.py`**

- **动作**: 读取并分析 `scripts/data/synthetic/niah.py` 的源代码。
- **目的**:
    - 理解其数据源（Paul Graham 的文章）。
    - 学习其上下文（haystack）的构建方法。
    - 确定“needle”是如何定义和插入的。
    - 熟悉其输出格式（JSONL）。

#### **第二步：创建新脚本文件 `configurable_niah.py`**

- **动作**: 在 `scripts/data/synthetic/` 目录下创建一个新文件 `configurable_niah.py`。
- **目的**:
    - 复制 `niah.py` 的基本结构作为新脚本的起点。
    - 避免直接修改原始脚本，保持项目原有结构。

#### **第三步：在新脚本中实现核心逻辑**

1.  **添加命令行参数**:
    - **动作**: 使用 Python 的 `argparse` 库在脚本开头定义 `--max-length`, `--length-interval`, 和 `--needle-positions` 三个参数。
    - **目的**: 允许用户从命令行灵活控制数据生成的细节。

2.  **实现循环生成逻辑**:
    - **动作**:
        - 编写一个外层循环，遍历从 `length-interval` 到 `max-length` 的所有长度，步长为 `length-interval`。
        - 在外层循环内部，再嵌套一个内层循环，从 1 迭代到 `needle-positions`。
        - 在内层循环中，计算“needle”插入的深度百分比，公式为 `depth_percent = i / needle_positions` (其中 `i` 是内层循环的当前值)。
    - **目的**: 根据用户输入的参数，系统地生成所有需要的上下文长度和“needle”插入位置的组合。

3.  **复用并修改数据处理功能**:
    - **动作**:
        - 复用 `niah.py` 中加载 Paul Graham 文章文本的函数。
        - 修改“needle”插入函数，使其接受 `depth_percent` 作为参数，并根据此百分比计算精确的插入位置。
        - 复用生成 JSONL 行并将其写入输出文件的函数。
    - **目的**: 最大化代码复用，将重点放在实现新的可配置逻辑上。

#### **第四步：最终确认和测试**

- **动作**:
    - 运行脚本并使用一些较小的参数值（例如 `--max-length 4000 --length-interval 1000 --needle-positions 4`）进行测试。
    - 检查生成的 JSONL 文件，确保其格式正确，并且“needle”已按预期插入到各个位置。
- **目的**: 保证脚本的正确性和健壮性。
