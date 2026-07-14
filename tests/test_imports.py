"""烟雾测试：M1 骨架阶段验证 import 与 config 读取。

不依赖 GPU、不依赖 COCO 数据、不依赖预训练权重下载。
运行：
    python tests/test_imports.py

也可用 pytest：
    pytest tests/test_imports.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_package_imports():
    import lead_net
    from lead_net.utils import load_config, resolve_paths_in, get_nested, project_root
    _ = lead_net.__version__
    assert project_root().exists()


def test_config_load():
    from lead_net.utils import load_config, get_nested
    cfg = load_config("configs/baseline_ssd.yaml")
    # 使用 lca 锁死为 false
    assert get_nested(cfg, "model.lca.enabled") is False
    # 类别数与 class_map 一致
    assert cfg["num_classes"] == len(cfg["class_map"])
    # inherit 合并：default 的 device 段应被保留
    assert "device" in cfg
    # paths 段应存在
    assert "paths" in cfg


def test_path_helpers():
    from lead_net.utils import project_root, resolve_path, ensure_dir
    root = project_root()
    assert root.is_dir()
    p = resolve_path("configs/default.yaml")
    assert p.exists()


def main():
    test_package_imports()
    test_config_load()
    test_path_helpers()
    print("[OK] M1 骨架烟雾测试通过：import / config / path 全部正常。")


if __name__ == "__main__":
    main()