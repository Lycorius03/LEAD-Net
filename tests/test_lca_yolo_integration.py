"""LCA × ultralytics 集成回归测试（Stage 1C 云端训练前置验证）。

覆盖 2026-07-18 发现的三个训练有效性 bug：
    1. alpha 死锁：训练模式下 forward 短路（abs(alpha)<1e-4 return identity）
       导致 alpha 无梯度、永远为 0，LCA 全程恒等 → 训练时必须走全图
    2. reduction 未传入：YAML `LCA, [r]` 的 r 被解析为 channels（LCA 签名首参），
       reduction 恒为默认 16；且 mip=max(8, 64//r) 在 P3 实际 64 通道下
       对 r=8/16/32 全部为 8 → r 消融三变体结构相同
    3. Detect P3 分支 from=16 绕过 LCA(17) → 小目标检测头吃不到 LCA 输出

同时保护既有性质（2026-07-17 修复）：alpha=0 时 LCA 初始输出恒等，
不破坏预训练特征；eval 模式下 alpha≈0 仍走短路省算力。

运行（需 torchenv：torch + ultralytics）：
    F:/.anaconda/envs/torchenv/python.exe tests/test_lca_yolo_integration.py
    pytest tests/test_lca_yolo_integration.py
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch

from lead_net.models.attention.lca import LCA

# 导入即注册 LCA 到 ultralytics（与 tools/cloud_train.py 相同路径）
from lead_net.models.yolo import lca_adapter  # noqa: F401
from ultralytics import YOLO

LCA_LAYER_INDEX = 17  # 与 tools/cloud_train.py 的 LCA_INSERT_INDEX 一致
_VARIANT_YAMLS = {
    8: _REPO_ROOT / "lead_net/models/yolo/yamls/yolo11n_lca_neck_r8.yaml",
    16: _REPO_ROOT / "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml",
    32: _REPO_ROOT / "lead_net/models/yolo/yamls/yolo11n_lca_neck_r32.yaml",
}
_P3_CHANNELS = 64  # YOLO11n width=0.25 下 P3 C3k2 输出 256*0.25

_model_cache: dict[int, YOLO] = {}


def _build_variant(reduction: int) -> YOLO:
    """构建 LCA 变体（YOLO 构造时的 stride dummy forward 已触发 LCA lazy 建层）。"""
    if reduction not in _model_cache:
        _model_cache[reduction] = YOLO(str(_VARIANT_YAMLS[reduction]), verbose=False)
    return _model_cache[reduction]


def _get_lca(model: YOLO) -> LCA:
    layer = model.model.model[LCA_LAYER_INDEX]
    assert isinstance(layer, LCA), f"层 {LCA_LAYER_INDEX} 应为 LCA，实际 {type(layer).__name__}"
    return layer


# ---------------------------------------------------------------- bug 1: alpha 死锁

def test_alpha_receives_grad_in_training():
    """训练模式下 alpha 必须参与计算图并收到非零梯度，否则永远无法离开 0。"""
    torch.manual_seed(42)
    lca = LCA(channels=_P3_CHANNELS, reduction=16)
    lca.train()
    # requires_grad 模拟真实训练中来自上游层的特征（有梯度流）
    x = torch.randn(2, _P3_CHANNELS, 8, 8, requires_grad=True)
    y = lca(x)
    y.sum().backward()
    assert lca.alpha.grad is not None, "训练模式下 alpha 无梯度（forward 短路死锁）"
    assert lca.alpha.grad.abs().item() > 0, "alpha 梯度为 0，无法脱离初始值"


def test_alpha_escapes_zero_with_sgd():
    """模拟真实训练：SGD 若干步后 alpha 必须离开 0（否则 LCA 全程恒等 = baseline）。"""
    torch.manual_seed(42)
    lca = LCA(channels=_P3_CHANNELS, reduction=16)
    lca.train()
    opt = torch.optim.SGD(lca.parameters(), lr=0.01, momentum=0.9)
    for _ in range(3):
        opt.zero_grad()
        x = torch.randn(2, _P3_CHANNELS, 8, 8, requires_grad=True)
        loss = (lca(x) - 1.0).pow(2).mean()
        loss.backward()
        opt.step()
    assert lca.alpha.abs().item() > 0, "SGD 3 步后 alpha 仍为 0：LCA 死锁为恒等层"


# ---------------------------------------------------- bug 2: reduction 传参 + mip 消融

def test_reduction_propagates_from_yaml():
    """YAML `LCA, [r]` 的 r 必须落到 LCA.reduction（而非被当作 channels）。"""
    for r in (8, 16, 32):
        lca = _get_lca(_build_variant(r))
        assert lca.reduction == r, (
            f"yolo11n_lca_neck_r{r}.yaml 期望 reduction={r}，实际 {lca.reduction}"
            "（YAML 参数被解析为 channels，reduction 用了默认值）"
        )


def test_mip_differs_across_reductions():
    """r=8/16/32 三个消融变体的中间通道 mip 必须互不相同（P3 实际 64 通道）。

    期望 mip = max(2, 64 // r) → r8:8, r16:4, r32:2。
    旧下限 max(8, ·) 会把三者全部压到 8，消融失去变量。
    """
    expected = {8: 8, 16: 4, 32: 2}
    actual = {}
    for r in (8, 16, 32):
        lca = _get_lca(_build_variant(r))
        assert lca._real_channels == _P3_CHANNELS, (
            f"LCA 实际建层通道应为 {_P3_CHANNELS}，实际 {lca._real_channels}"
        )
        actual[r] = lca.mip
    assert actual == expected, f"mip 期望 {expected}，实际 {actual}（消融变体结构相同）"


# ------------------------------------------------------ bug 3: Detect P3 接 LCA 输出

def test_detect_p3_consumes_lca_output():
    """Detect 的 P3 分支必须接 LCA(17) 的输出而非 LCA 之前的层 16。"""
    for r in (8, 16, 32):
        model = _build_variant(r)
        detect = model.model.model[-1]
        assert detect.f == [17, 20, 23], (
            f"yolo11n_lca_neck_r{r}.yaml Detect from 期望 [17, 20, 23]，"
            f"实际 {detect.f}（P3 检测头绕过了 LCA）"
        )


# ------------------------------------------- bug 4: train() 丢弃重映射权重

def test_remapper_sets_ckpt_for_train_handoff():
    """load_pretrained_with_remapping 返回的模型必须带非空 ckpt。

    ultralytics model.train()（engine/model.py:808）只有在 self.ckpt 为
    truthy 时才把当前内存权重交给 trainer；否则 trainer 用 yaml 冷启动
    随机初始化训练 —— 重映射的 448 项预训练权重被整体丢弃
    （云端 smoke 实测 lca_r16 mAP@0.5=0.0011 vs baseline 0.216）。
    """
    from lead_net.models.yolo.weight_remapper import load_pretrained_with_remapping

    model = load_pretrained_with_remapping(
        str(_REPO_ROOT / "lead_net/models/yolo/yamls/yolo11n_lca_neck_r16.yaml"),
        str(_REPO_ROOT / "yolo11n.pt"), lca_insert_index=17, verbose=False,
    )
    assert bool(model.ckpt), (
        "self.ckpt 为空：model.train() 不会复用重映射权重，"
        "trainer 将从随机初始化开始训练"
    )
    # 不得误触发 resume 逻辑（engine/model.py:797 检查 epoch>=0 且有 optimizer）
    assert model.ckpt.get("epoch", -1) < 0, "ckpt 不应携带可 resume 的 epoch"
    assert model.ckpt.get("optimizer") is None, "ckpt 不应携带 optimizer 状态"


# ------------------------------------------------- 回归保护：初始恒等性 & eval 短路

def test_initial_forward_is_identity_in_training():
    """alpha=0 时训练模式输出必须严格等于输入（保护预训练特征，2026-07-17 修复）。"""
    torch.manual_seed(42)
    lca = LCA(channels=_P3_CHANNELS, reduction=16)
    lca.train()
    x = torch.randn(2, _P3_CHANNELS, 8, 8)
    y = lca(x)
    assert torch.equal(y, x), "alpha=0 时 LCA 初始输出应严格恒等于输入"


def test_eval_shortcut_preserved():
    """eval 模式且 alpha≈0 时仍应短路直接返回输入张量（部署省算力路径）。"""
    torch.manual_seed(42)
    lca = LCA(channels=_P3_CHANNELS, reduction=16)
    lca.eval()
    x = torch.randn(2, _P3_CHANNELS, 8, 8)
    with torch.no_grad():
        y = lca(x)
    assert y.data_ptr() == x.data_ptr(), "eval 模式 alpha≈0 应短路返回原张量"


def main():
    test_alpha_receives_grad_in_training()
    test_alpha_escapes_zero_with_sgd()
    test_reduction_propagates_from_yaml()
    test_mip_differs_across_reductions()
    test_detect_p3_consumes_lca_output()
    test_remapper_sets_ckpt_for_train_handoff()
    test_initial_forward_is_identity_in_training()
    test_eval_shortcut_preserved()
    print("[OK] LCA × ultralytics 集成测试通过：alpha 可学习、reduction 消融有效、Detect P3 接 LCA。")


if __name__ == "__main__":
    main()
