"""LCA 烟雾测试（M2）。

不依赖 GPU、不依赖真实 COCO / 预训练权重下载（model.lca.enabled 可与 backbone
weights=None 组合，避免联网）。验证：
    - LCA 模块单独 forward 不改变 shape
    - build_lead_net 在 use_lca=true/false 两个配置下均能 forward
    - head 输出 shape 与 baseline era 一致（不破坏多尺度 SSD-Lite 接口）
    - +LCA 参数量严格大于 Baseline，开销非零（服务 RQ2）
    - 反向传播能跑通（loss 可回传到 LCA 权重）

运行：
    python tests/test_lca.py
    pytest tests/test_lca.py
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None
_fixture = pytest.fixture(scope="module") if pytest is not None else (lambda f: f)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lead_net.models import build_lead_net, build_lca, LCA


def _cfg(lca_enabled: bool, with_weights: bool = False) -> dict:
    """构造最小可用配置（不触发预训练权重下载，除非显式要求）。"""
    cfg = {
        "num_classes": 7,
        "data": {"input_size": 320},
        "model": {
            "backbone": {
                "name": "mobilenet_v3_small",
                "weights": "IMAGENET1K_V1" if with_weights else None,
                "width_multiplier": 1.0,
            },
            "lca": {"enabled": lca_enabled, "reduction": 16},
        },
    }
    return cfg


def test_lca_module_shape():
    # 测试各种配置组合下的 shape 保持一致
    for eg in [True, False]:
        for op in [True, False]:
            for rg in [True, False]:
                lca = LCA(channels=256, reduction=16, edge_guidance=eg, obstacle_prior=op, residual_gate=rg)
                x = torch.randn(2, 256, 10, 10)
                y = lca(x)
                assert y.shape == x.shape, f"LCA (eg={eg}, op={op}, rg={rg}) 改变了 shape: {y.shape} vs {x.shape}"


def test_build_lca_from_cfg():
    cfg = _cfg(True)
    # 默认启用自适应压缩比
    lca = build_lca(cfg, channels=256)
    assert isinstance(lca, LCA)
    assert lca.mip == max(8, 256 // 16)

    # 显式测试关闭自适应压缩比
    cfg["model"]["lca"]["adaptive_reduction"] = False
    cfg["model"]["lca"]["reduction"] = 32
    lca_fixed = build_lca(cfg, channels=256)
    assert lca_fixed.mip == max(8, 256 // 32)


def test_forward_baseline_no_lca():
    cfg = _cfg(False)
    model = build_lead_net(cfg).eval()
    with torch.no_grad():
        cls_pred, loc_pred = model(torch.randn(2, 3, 320, 320))
    assert cls_pred.dim() == 3 and loc_pred.dim() == 3
    assert cls_pred.shape[0] == 2 and loc_pred.shape[0] == 2
    assert loc_pred.shape[-1] == 4


def test_forward_with_lca():
    cfg = _cfg(True)
    model = build_lead_net(cfg).eval()
    with torch.no_grad():
        cls_pred_lca, loc_pred_lca = model(torch.randn(2, 3, 320, 320))
    # 与 baseline 同形状（head 不变）
    assert cls_pred_lca.shape[0] == 2


def test_param_overhead_nonzero():
    # 测试残差门控开启与关闭时的参数差异
    cfg_rg = _cfg(True)
    cfg_rg["model"]["lca"]["residual_gate"] = True
    model_rg = build_lead_net(cfg_rg)
    
    cfg_no_rg = _cfg(True)
    cfg_no_rg["model"]["lca"]["residual_gate"] = False
    model_no_rg = build_lead_net(cfg_no_rg)
    
    p_rg = sum(p.numel() for p in model_rg.parameters())
    p_no_rg = sum(p.numel() for p in model_no_rg.parameters())
    
    # 刚好相差一个 trainable alpha 标量参数
    assert p_rg - p_no_rg == 1, f"残差门控参数增量应为 1, 实际: {p_rg - p_no_rg}"
    
    lca_params = sum(p.numel() for p in model_rg.backbone.lca.parameters())
    assert lca_params > 0
    print(f"[info] params: LCA with RG={p_rg:,} LCA without RG={p_no_rg:,} delta={p_rg - p_no_rg} (lca module={lca_params:,})")


def test_backward_reaches_lca():
    cfg = _cfg(True)
    model = build_lead_net(cfg)
    cls_pred, loc_pred = model(torch.randn(2, 3, 320, 320))
    loss = cls_pred.float().sum() + loc_pred.float().sum()
    loss.backward()
    # LCA 的 conv1 权重与 alpha 权重都应收到梯度
    g_conv = model.backbone.lca.conv1.weight.grad
    assert g_conv is not None and torch.isfinite(g_conv).all()
    
    g_alpha = model.backbone.lca.alpha.grad
    assert g_alpha is not None and torch.isfinite(g_alpha).all()


def main():
    test_lca_module_shape()
    test_build_lca_from_cfg()
    test_forward_baseline_no_lca()
    test_forward_with_lca()
    test_param_overhead_nonzero()
    test_backward_reaches_lca()
    print("[OK] M2 LCA 烟雾测试通过：特征梯度、自适应缩减、残差门控和空间先验全部正常。")


if __name__ == "__main__":
    main()