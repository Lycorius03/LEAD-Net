"""配置加载工具。

依据 docs/ARCHITECTURE.md §配置驱动：
    - 所有超参数通过 configs/*.yaml 传入，代码中不允许硬编码。
    - 支持任务配置通过 `inherit:` 字段继承公共配置（深度合并）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .path import PathLike, resolve_path


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 优先于 base。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: PathLike) -> dict:
    """加载单个 yaml 为 dict。"""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(path: PathLike) -> dict:
    """加载配置，自动处理 `inherit:` 字段（递归）。

    若配置含 `inherit: other.yaml`，先把 other 加载为基础，再用本配置覆盖合并。
    inherit 路径相对于本配置文件所在目录解析。
    """
    config_path = resolve_path(path)
    raw = load_yaml(config_path)
    inherit_name = raw.pop("inherit", None)
    if inherit_name:
        inherit_path = (config_path.parent / inherit_name).resolve()
        base = load_config(inherit_path)
        return _deep_merge(base, raw)
    return raw


def get_nested(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """按点号路径取嵌套值，缺失返回 default。

    例：get_nested(cfg, "model.lca.enabled", False)。
    """
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_paths_in(cfg: dict) -> dict:
    """递归解析 cfg["paths"] 下所有相对路径为绝对 Path 对象。

    只处理顶层 paths 段，其余段保持原值。
    """
    out = copy.deepcopy(cfg)
    paths = out.get("paths")
    if isinstance(paths, dict):
        out["paths"] = {k: (resolve_path(v) if v is not None else None)
                        for k, v in paths.items()}
    return out