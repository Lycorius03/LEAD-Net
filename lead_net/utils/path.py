"""路径工具（强制 pathlib，跨平台适配）。

依据 docs/ARCHITECTURE.md §跨平台适配规范：
    - 一律使用 pathlib.Path，禁止字符串拼接路径（含 os.path.join）。
    - 代码中不写死绝对路径，所有路径通过 configs/*.yaml 传入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def project_root() -> Path:
    """返回项目根目录。

    本模块位于 <root>/lead_net/utils/path.py，向上两级即项目根。
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: PathLike, base: PathLike | None = None) -> Path:
    """解析相对/绝对路径为绝对 Path。

    Args:
        path: 配置文件中给出的相对/绝对路径。
        base: 相对路径基准目录；默认为项目根。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base_dir = Path(base) if base is not None else project_root()
    return (base_dir / p).resolve()


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在，不存在则创建；返回绝对 Path。"""
    p = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p