可控长度/间隔/插入位置数量的 NIAH 扩展脚本计划

目标
- 基于现有 `niah.py` 增强，支持（按 tokenizer token 计数）：
  - 可控总长度（如 32000、128000）。
  - 可控间隔/分段长度（如 1000、2000）。
  - 可控每段插入 needle 的位置数量（如 5、10），按等比例百分位（10%、20% …）。
  - 可选：全局随机偏移/抖动、插入冲突处理、上下文/needle 模板化。

输出
- 生成的数据写入：`benchmark_root/<MODEL>/<BENCHMARK>/<SEQ>/data`，与现有管线兼容。
- 预测写入：`benchmark_root/<MODEL>/<BENCHMARK>/<SEQ>/pred`（沿用现有推理与评估脚本）。

接口与文件组织
- 新脚本：`scripts/data/synthetic/niah_variable.py`
  - 仅依赖标准库 + 项目已有工具；遵循 PEP 8。
  - 保持与 `niah.py` 近似的输入/输出字段：`index`, `input`, `outputs`（ground truth）、并在生成阶段不产生 `pred`。
  - 复用 `scripts/data/synthetic/niah.py` 中的文本/needle 模板与指令框架，确保与现有评估一致。
- 可选配置文件（保持简单，优先命令行参数）：`--config` 支持 JSON/YAML。

命令行参数（建议）
- `--total_length`/`-L`：总长度（int），如 128000。
- `--segment_length`/`-S`：分段长度（int），如 1000。
- `--positions_per_segment`/`-P`：每段插入位置数量（int），如 10。
- `--positions_mode`：`percent`（默认，等比例：避开 0%/100%）或 `linear`（等距点位，避开边界）。
- `--needle`：needle 模板或从模板集合中采样的 key。
- `--jitter_ratio`：位置抖动比例（0-0.5），默认 0（无抖动）。
- `--seed`：随机种子，默认 0。
- `--num_samples`：样本数量（与 `config_tasks.sh` 对齐）。
- `--output`：输出 JSONL 路径（默认写入 `benchmark_root/.../data`）。
- `--tokenizer`：Hugging Face tokenizer 名称或本地路径（必填，用于 token 级长度控制）。
- `--trust_remote_code`：是否信任自定义 tokenizer 代码，默认 `false`。
- `--final_length_mode`：`base`（默认，基底长度为 `total_length`，插入后可能略超）或 `strict`（预留插入余量，最终长度尽量贴合 `total_length`）。

生成逻辑（核心算法，token 级）
1) 用 `--tokenizer` 对文本进行分词计数，所有长度与位置均以 tokens 为单位。
2) 计算分段数：`num_segments = total_length // segment_length`（允许余数段可选丢弃/拼接）。
3) 对每个样本：
   - 基底 haystack 生成：
     - final_length_mode = `base`：先合成足量自然文本，截取前 `total_length` tokens 作为基底。
     - final_length_mode = `strict`：估算 needle token 长度的期望值 E，先构造 `total_length - num_segments*P*E` 作为基底，再在插入后按需微调（轻度截断/填充）以贴合 `total_length`。
   - 段与位置：对每个段 `i in [0, num_segments)`：
     - 位置计算（避开边界）：`pos_k = floor(segment_length * (k+1)/(P+1))`，k=0..P-1；等价于等比例分位，天然避开 0%/100%。
     - 抖动：`delta = round(U(-jitter_ratio, +jitter_ratio) * segment_length)`，并裁剪使得 `1 ≤ pos_k+delta ≤ segment_length-1`。
     - 全局 token 索引：`insert_idx = i*segment_length + (pos_k+delta)`。
   - 冲突处理：如多个插入索引相等或跨越已插入区域，按“顺延至最近可用位置但不越段”的策略；若无法顺延则跳过并在 meta 中记录。
   - 执行插入：将 needle 文本插入到 `insert_idx` 对应的 token 边界上，更新后续索引偏移。
   - 记录 ground truth：段索引、相对/绝对 token 位置、needle 文本/ID。
4) 写出 JSONL，字段稳定。

与现有管线集成
- 在 `scripts/config_tasks.sh` 中添加变量：
  - `SEQ`（如 128k），`TOTAL_LENGTH`, `SEGMENT_LENGTH`, `POSITIONS_PER_SEG`, `NUM_SAMPLES`。
  - 为 synthetic 任务追加一个 case，将上述参数传入 `niah_variable.py`。
- 在 `scripts/run.sh` 的 synthetic 分支中，检测是否使用 `niah_variable.py`，并将参数透传。
- 评估沿用 `scripts/eval/evaluate.py`（无需改动）。

验证与最小实验
- 快速本地验证：
  - 将 `NUM_SAMPLES` 暂降到 2-5；
  - 设置 `--total_length 32000 --segment_length 1000 --positions_per_segment 5 --tokenizer <tok>`；
  - 校验：
    - 每条 `input` 的 token 长度（base 模式下≈`total_length`，strict 模式下≈严格等于）。
    - 插入计数≈`num_segments * P`，位置均避开边界且位于对应段内。
- 运行最小评估：`python scripts/eval/evaluate.py --data_dir benchmark_root/<MODEL>/synthetic/<SEQ>/pred --benchmark synthetic`。

边界与鲁棒性处理
- 总长度非 `segment_length` 整数倍：
  - 方案 A：忽略尾段（默认）。
  - 方案 B：尾段按比例计算位置但不满配（可加 `--keep_tail_segment`）。
- 插入冲突：
  - 方案 A：顺延到下一个空位（不越段）。
  - 方案 B：该位置跳过并计数（记录到 meta 方便调试）。
- 长度校验（token 级）：
  - base 模式：最终 tokens 可能略超 `total_length`（≈`total_length + Σ|needle|_tok`）。
  - strict 模式：最终 tokens ≈ `total_length`，必要时末尾轻度截断/填充无信息 token（如空白/换行）。

开发步骤（执行顺序）
1) 阅读与梳理 `scripts/data/synthetic/niah.py` 的输入输出和模板调用点。
2) 拆出共用工具（如随机、模板、写 JSONL）到局部 helper，避免重复逻辑。
3) 实现 `niah_variable.py` 的参数解析与位置生成器（percent/linear + jitter）。
4) 实现插入器：处理冲突、边界、索引映射和 ground truth 记录。
5) 接入现有数据输出目录结构，保持字段一致。
6) 在 `config_tasks.sh`/`run.sh` 中追加分支与环境变量透传。
7) 本地用小规模样本做 sanity check，核对长度、插入计数与位置分布。
8) 运行一次端到端（小样本）并生成 `summary*.csv`，确认评估可跑通。
9) 调整文档与 README 片段（如有示例命令变更）。

示例命令
- 128k 总长、每段 1000、每段 10 个位置（等比例分位，避开边界）：
  - `python scripts/data/synthetic/niah_variable.py \
     --total_length 128000 --segment_length 1000 \
     --positions_per_segment 10 --positions_mode percent \
     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
     --num_samples 10 --seed 42`
- 32k 总长、每段 2000、每段 5 个位置，等距避开边界 + 抖动 5%（strict 模式）：
  - `python scripts/data/synthetic/niah_variable.py \
     --total_length 32000 --segment_length 2000 \
     --positions_per_segment 5 --positions_mode linear \
     --jitter_ratio 0.05 --final_length_mode strict \
     --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
     --num_samples 4`

里程碑与验收
- M1：脚本完成 + 单元级位置分布校验（1 天）。
- M2：接入 run.sh/config_tasks.sh，最小样本跑通（0.5 天）。
- M3：长序列（≥128k）端到端评估通过，生成 summary（0.5 天）。

注意事项
- 遵循项目键名与路径规范；不提交任何密钥；
- 生成文本与 needle 模板保持可复现（固定 seed）；
- 长序列内存峰值控制：使用列表拼接/`io.StringIO`，避免 O(n^2) 复制；
- 默认使用 tokenizer 级长度与位置控制；`--tokenizer` 为必填。
