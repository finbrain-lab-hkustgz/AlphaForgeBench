"""
Query Batch Generator - 批量生成策略需求Query
为9个类别各生成策略需求query

特点：
1. 同一类别内的query尽量不重复
2. 不同类别之间也不要太相似
3. 支持断点续传
4. 所有配置从 generation_config.json 读取
"""
import httpx
import json
import asyncio
import os
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from json_repair import repair_json

from dotenv import load_dotenv
load_dotenv(verbose=True)


# ============ 配置文件路径 ============
CONFIG_FILE = Path(__file__).parent / "configs" / "generation_config.json"


# ============ 全局配置（从JSON加载） ============
class GenerationConfig:
    """从JSON文件加载的配置"""
    _instance = None
    _loaded = False

    # 默认值
    model: str = ""
    target_counts_default: int = 20
    target_counts: Dict[str, int] = {}
    level_definitions: Dict[str, Dict[str, str]] = {}
    base_prompt: str = ""
    category_definitions: Dict[str, str] = {}
    style_definitions: str = ""

    # 运行时配置
    OUTPUT_DIR: Path = None

    @classmethod
    def load(cls, config_path: Path = CONFIG_FILE):
        """从JSON文件加载配置"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cls.model = data.get("model", "")
        cls.target_counts_default = data.get("target_counts_default", 20)
        cls.target_counts = data.get("target_counts", {})
        cls.level_definitions = data.get("prompts", {}).get("level_definitions", {})
        cls.base_prompt = data.get("prompts", {}).get("base_prompt", "")
        cls.category_definitions = data.get("prompts", {}).get("category_definitions", {})
        cls.style_definitions = data.get("prompts", {}).get("style_definitions", "")
        cls._loaded = True

        return cls

    @classmethod
    def get_categories(cls) -> List[str]:
        """获取所有类别"""
        return list(cls.target_counts.keys())

    @classmethod
    def get_target_count(cls, category: str) -> int:
        """获取指定类别的目标数量，-1 表示使用默认值"""
        count = cls.target_counts.get(category, -1)
        if count < 0:
            return cls.target_counts_default
        return count

    @classmethod
    def get_total_target(cls) -> int:
        """获取总目标数量"""
        return sum(cls.get_target_count(cat) for cat in cls.target_counts.keys())


class RuntimeConfig:
    """运行时固定配置"""
    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    BATCH_SIZE = 3     # 每次生成数量（对应3种风格）
    MAX_RETRIES = 3    # 最大重试次数
    TIMEOUT = 300      # 请求超时（秒）
    REQUEST_DELAY = 1  # 请求间隔（秒）
    MAX_CONCURRENT = 5 # 最大并发数


# ============ 配置Hash与目录管理 ============
def compute_config_hash() -> str:
    """计算当前配置的hash，用于判断是否可以续传"""
    config_str = json.dumps({
        "model": GenerationConfig.model,
        "level_definitions": GenerationConfig.level_definitions,
        "base_prompt": GenerationConfig.base_prompt,
        "category_definitions": GenerationConfig.category_definitions,
        "style_definitions": GenerationConfig.style_definitions,
        "target_counts": GenerationConfig.target_counts
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def copy_config_to_output(output_dir: Path):
    """复制输入配置文件到输出目录"""
    dest = output_dir / "generation_config.json"
    shutil.copy2(CONFIG_FILE, dest)

    # 添加运行时信息
    with open(dest, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["_runtime"] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": compute_config_hash()
    }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def find_or_create_output_dir() -> Path:
    """查找匹配的目录或创建新目录"""
    base_dir = Path(__file__).parent / "generated_queries"
    base_dir.mkdir(parents=True, exist_ok=True)

    current_hash = compute_config_hash()

    # 扫描现有目录，查找匹配的配置
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            config_file = subdir / "generation_config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    if config.get("_runtime", {}).get("config_hash") == current_hash:
                        print(f"[续传] 找到匹配目录: {subdir.name}")
                        return subdir
                except Exception:
                    continue

    # 没有找到匹配目录，创建新目录
    model_name = GenerationConfig.model.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = base_dir / f"{model_name}_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)

    # 复制配置文件到输出目录
    copy_config_to_output(new_dir)
    print(f"[新建] 创建目录: {new_dir.name}")
    return new_dir


def get_output_paths() -> Dict[str, Path]:
    """获取输出文件路径"""
    if GenerationConfig.OUTPUT_DIR is None:
        GenerationConfig.OUTPUT_DIR = find_or_create_output_dir()
    return {
        "output": GenerationConfig.OUTPUT_DIR / "generated_queries.json",
        "checkpoint": GenerationConfig.OUTPUT_DIR / "checkpoint.json",
        "config": GenerationConfig.OUTPUT_DIR / "generation_config.json",
    }


# ============ Prompt Builder ============
def get_level_overview() -> str:
    """Get overview of all three levels (L1/L2/L3)"""
    lines = ["## Level Overview (Understanding the Hierarchy)\n"]
    lines.append("The three levels represent a spectrum from explicit to abstract:\n")

    for level_key in ["L1", "L2", "L3"]:
        level_def = GenerationConfig.level_definitions.get(level_key, {})
        name = level_def.get("name", level_key)
        core_idea = level_def.get("core_idea", "")
        description = level_def.get("description", "")
        lines.append(f"### {name}")
        lines.append(f"**Core Idea**: {core_idea}")
        lines.append(f"{description}\n")

    return "\n".join(lines)


def get_all_categories_full(current_category: str) -> str:
    """Get full definitions of all categories for reference"""
    lines = ["## All Category Definitions (for reference)\n"]

    for cat, definition in GenerationConfig.category_definitions.items():
        if cat == current_category:
            lines.append(f"### {cat} ← (CURRENT TASK - see detailed requirements below)")
        else:
            lines.append(f"### {cat}")
        lines.append(definition)
        lines.append("")

    return "\n".join(lines)


def get_current_task_emphasis(category: str) -> str:
    """Get emphasized current task requirements"""
    # Extract level (L1, L2, L3) from category
    level_key = category.split("_")[0]  # e.g., "L1_easy" -> "L1"
    level_def = GenerationConfig.level_definitions.get(level_key, {})
    category_def = GenerationConfig.category_definitions.get(category, "")

    level_name = level_def.get("name", level_key)
    core_idea = level_def.get("core_idea", "")
    level_description = level_def.get("description", "")

    emphasis = f"""
## !!! CURRENT TASK: {category} !!!

### Level Requirement: {level_name}
**Core Idea**: {core_idea}
{level_description}

### Specific Requirements for {category}:
{category_def}

### CRITICAL REMINDERS:
- Your generated query MUST strictly follow the {level_key} constraints above
- {"Include EXACT column names (e.g., df['rsi_14']) AND specific numeric values" if level_key == "L1" else ""}
- {"Indicate factor types but leave specific values to be inferred" if level_key == "L2" else ""}
- {"Provide only objectives and constraints - NO specific factors or values" if level_key == "L3" else ""}
"""
    return emphasis


def build_prompt(category: str, existing_summaries: List[str], batch_size: int = 3) -> str:
    """Build generation prompt with deduplication context (using summaries only)"""

    # Get all prompt components
    level_overview = get_level_overview()
    all_categories = get_all_categories_full(category)
    current_task = get_current_task_emphasis(category)

    # Deduplication context - use summaries instead of full queries to prevent leakage
    if existing_summaries:
        existing_context = "\n".join([f"- {s}" for s in existing_summaries[-15:] if s])  # Show last 15
        dedup_instruction = f"""
## Already Generated Strategies (DO NOT REPEAT similar ideas)
{len(existing_summaries)} strategies already generated. Here are recent summaries:
{existing_context}

**Deduplication Requirements:**
- Use different factor combinations
- Use different trading logic
- Avoid strategies that are conceptually similar to the above
"""
    else:
        dedup_instruction = ""

    prompt = f"""{GenerationConfig.base_prompt}

{level_overview}

{all_categories}

{current_task}

{GenerationConfig.style_definitions}

{dedup_instruction}

## Output Requirements
1. Generate exactly 3 strategy objects - one for each style (Conservative, Aggressive, Balanced)
2. Each query should be a complete strategy requirement description (3-8 sentences)
3. **PROFITABILITY IS THE GOAL** - every strategy must be designed with profit potential in mind
4. **SIGNAL GENERATION**: Use factor thresholds (e.g., rsi_14 < 30) OR factor comparisons (e.g., ema_12 > ema_26) to generate signals
5. **THRESHOLD VALIDITY (L1 only)**: For explicit threshold strategies, ensure values are within the factor's actual range:
   - rank_20, imax_20, imin_20: range is 0 to 0.95 (NEVER reaches 1.0)
   - max_w: always ≥1.0 | min_w: always 0-1 | std_w, vstd_w: always ≥0
6. **DIVERSITY**: Vary factor combinations, logic patterns, and trading styles across strategies
7. **STRICTLY FOLLOW THE CURRENT LEVEL CONSTRAINTS** - this is the most important requirement
8. Include distinctive twists (time-based exit, partial scaling, volatility filter, etc.)
9. Keep it single-stock (no cross-sectional ranking)
10. **LANGUAGE DIVERSITY (CRITICAL)**: Each query MUST have a distinctly different writing style:
    - Vary sentence openers: "The strategy...", "When...", "Buy when...", "This approach...", "Enter long if...", etc.
    - Vary sentence lengths: mix short punchy sentences with longer detailed ones
    - Vary structure: some queries start with entry conditions, others with exit logic, others with the overall goal
    - Do not use the same sentence pattern for all 3 queries in a batch

## Output Format (STRICT)
Return ONLY a JSON array. Each item must have:
- "style": one of "conservative", "aggressive", "balanced"
- "summary": A single sentence (10-20 words) summarizing the strategy's core idea
- "query": The full strategy requirement description

```json
[
    {{"style": "conservative", "summary": "Brief strategy description", "query": "..."}},
    {{"style": "aggressive", "summary": "Brief strategy description", "query": "..."}},
    {{"style": "balanced", "summary": "Brief strategy description", "query": "..."}}
]
```
"""
    return prompt


# ============ API调用和解析 ============
def parse_queries(content: str) -> List[Dict[str, str]]:
    """解析LLM响应，提取query列表（带style, summary, query）"""
    import re

    # 尝试提取JSON数组
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        arr_start = content.find("[")
        arr_end = content.rfind("]") + 1
        if arr_start != -1 and arr_end > arr_start:
            json_str = content[arr_start:arr_end]
        else:
            json_str = content.strip()

    # 解析JSON
    try:
        queries = json.loads(json_str)
    except json.JSONDecodeError:
        queries = json.loads(repair_json(json_str))

    if not isinstance(queries, list):
        raise ValueError("响应不是数组格式")

    # 处理结果
    result = []
    for item in queries:
        if isinstance(item, dict) and "query" in item:
            result.append({
                "style": item.get("style", "unknown"),
                "summary": item.get("summary", ""),
                "query": item.get("query", "").strip()
            })

    return result


async def generate_batch(
    category: str,
    existing_summaries: List[str],
    batch_size: int = 3
) -> List[Dict[str, str]]:
    """调用API生成一批query（带style和summary标签）"""

    prompt = build_prompt(category, existing_summaries, batch_size)

    async with httpx.AsyncClient(timeout=float(RuntimeConfig.TIMEOUT)) as client:
        for attempt in range(RuntimeConfig.MAX_RETRIES):
            try:
                response = await client.post(
                    f"{RuntimeConfig.OPENROUTER_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {RuntimeConfig.OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": GenerationConfig.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 32768,
                    }
                )
                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    raise ValueError(f"API错误: {result['error']}")

                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    return parse_queries(content)
                else:
                    raise ValueError("无法解析响应")

            except Exception as e:
                if attempt == RuntimeConfig.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    return []


# ============ 断点续传 ============
def load_checkpoint() -> Dict[str, List[Dict[str, str]]]:
    """加载断点（带style标签）"""
    paths = get_output_paths()
    checkpoint_path = paths["checkpoint"]

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 加载断点失败: {e}")

    return {cat: [] for cat in GenerationConfig.get_categories()}


def save_checkpoint(data: Dict[str, List[Dict[str, str]]]):
    """保存断点"""
    paths = get_output_paths()
    checkpoint_path = paths["checkpoint"]

    try:
        tmp_path = checkpoint_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, checkpoint_path)
    except Exception as e:
        print(f"[WARN] 保存断点失败: {e}")


# ============ 进度条 ============
def print_progress(category: str, current: int, total: int):
    """打印简洁进度"""
    bar_len = 20
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {current}/{total}", end="", flush=True)


# ============ 主循环 ============
async def process_category(
    category: str,
    queries_data: Dict[str, List[Dict[str, str]]]
) -> int:
    """处理单个类别，返回新增数量"""
    existing = queries_data.get(category, [])
    added = 0
    target_count = GenerationConfig.get_target_count(category)

    # 提取已有的summary用于去重
    existing_summaries = [item.get("summary", "") for item in existing if isinstance(item, dict)]

    print(f"\n{category}: {len(existing)}/{target_count}")

    while len(existing) < target_count:
        print_progress(category, len(existing), target_count)

        try:
            new_queries = await generate_batch(category, existing_summaries)

            # 去重：基于 summary 去重
            for item in new_queries:
                summary = item.get("summary", "")
                if summary and summary not in existing_summaries and len(existing) < target_count:
                    existing.append(item)
                    existing_summaries.append(summary)
                    added += 1

            # 更新并保存
            queries_data[category] = existing
            save_checkpoint(queries_data)

            # 请求间隔
            if len(existing) < target_count:
                await asyncio.sleep(RuntimeConfig.REQUEST_DELAY)

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            break

    print_progress(category, len(existing), target_count)
    print(f" ✓ (+{added})")

    return added


async def main():
    """主函数"""
    print("=" * 50)
    print("Query Batch Generator")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载配置文件
    try:
        GenerationConfig.load()
        print(f"配置文件: {CONFIG_FILE}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"模型: {GenerationConfig.model}")
    categories = GenerationConfig.get_categories()
    total_target = GenerationConfig.get_total_target()
    print(f"目标: {len(categories)}类别, 共{total_target}个")
    print()

    if not RuntimeConfig.OPENROUTER_API_KEY:
        print("[ERROR] 未设置 OPENROUTER_API_KEY")
        return

    # 初始化输出目录（根据配置hash查找或创建）
    GenerationConfig.OUTPUT_DIR = find_or_create_output_dir()
    print(f"输出目录: {GenerationConfig.OUTPUT_DIR}")
    print()

    # 加载断点
    queries_data = load_checkpoint()

    # 统计已有数量
    total_existing = sum(len(v) for v in queries_data.values())
    print(f"已有: {total_existing}个")

    # 处理每个类别（串行处理，因为去重需要依赖已生成的内容）
    total_added = 0
    for category in categories:
        added = await process_category(category, queries_data)
        total_added += added

    # 保存最终结果
    paths = get_output_paths()
    with open(paths["output"], "w", encoding="utf-8") as f:
        json.dump(queries_data, f, ensure_ascii=False, indent=2)

    # 统计
    print("\n" + "=" * 50)
    print("完成统计")
    print("=" * 50)
    for cat in categories:
        target = GenerationConfig.get_target_count(cat)
        print(f"  {cat}: {len(queries_data.get(cat, []))}/{target}")
    total = sum(len(v) for v in queries_data.values())
    print(f"\n总计: {total}个 (新增: {total_added})")
    print(f"输出: {paths['output']}")


if __name__ == "__main__":
    asyncio.run(main())
