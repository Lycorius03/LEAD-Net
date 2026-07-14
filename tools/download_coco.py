"""COCO 2017 下载脚本（resumable，支持 split 与 SHA256 校验）。

依据：
    - docs/DATASET.md：主数据集 COCO，锁定 2017 版。
    - docs/ARCHITECTURE.md §跨平台 4：产出物须与配置一起可复现；数据集校验和须记录。
    - docs/DATASET.md §数据集哈希/快照：下载时间与校验和需回填。

用法：
    python tools/download_coco.py --root data/coco --split all
    python tools/download_coco.py --root data/coco --split val2017 --split annotations
    python tools/download_coco.py --root data/coco --split train2017 --check

注：
    - train2017 约 18GB、val2017 约 1GB、annotations 约 251MB；
      大文件下载支持断点续传（HTTP Range）。
    - 下载完成后可选 --check 计算 zip 的 SHA256，结果打印并建议回填 DATASET.md。
    - 下载/解压后请运行 `python tools/inspect_data.py --config configs/baseline_ssd.yaml`
      做数据验证。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# COCO 2017 官方下载地址（与 docs/DATASET.md 锁定 2017 版一致）
URLS: dict[str, str] = {
    "train2017":     "http://images.cocodataset.org/zips/train2017.zip",
    "val2017":       "http://images.cocodataset.org/zips/val2017.zip",
    "annotations":   "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# block size for streaming download & hash
_CHUNK = 1024 * 1024  # 1 MiB

USER_AGENT = "LEAD-Net/downloader (Mozilla/5.0 compatible)"


def _remote_size(url: str) -> int | None:
    req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as r:  # noqa: S310 信任 COCO 官方源
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except (HTTPError, URLError, TimeoutError):
        return None


def _download_resumable(url: str, dst: Path) -> None:
    """断点续传下载到 dst。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    total = _remote_size(url)
    if total is not None and have == total:
        part.replace(dst)
        print(f"[skip] {dst.name} 已完整（{total} bytes）")
        return
    headers = {"User-Agent": USER_AGENT}
    if have > 0:
        headers["Range"] = f"bytes={have}-"
    req = Request(url, headers=headers)
    print(f"[get ] {url}\n       -> {part}  (已有 {have} bytes)")
    with urlopen(req, timeout=60) as r:  # noqa: S310
        if have > 0 and r.status == 200:
            # 服务器不支持续传：重新下载
            have = 0
        mode = "ab" if have > 0 else "wb"
        downloaded = have
        with open(part, mode) as f:
            while True:
                chunk = r.read(_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r       ... {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
            print()
    part.replace(dst)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def unzip_to(zip_path: Path, target_root: Path) -> None:
    print(f"[unzip] {zip_path} -> {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_root)
    print("[unzip] done")


def main() -> int:
    ap = argparse.ArgumentParser(description="COCO 2017 下载（resumable）")
    ap.add_argument("--root", type=str, default="data/coco", help="COCO 根目录")
    ap.add_argument(
        "--split",
        action="append",
        choices=["all", "train2017", "val2017", "annotations"],
        required=True,
        help="指定 split（可多次）",
    )
    ap.add_argument("--no-unzip", action="store_true", help="下载后不解压")
    ap.add_argument("--check", action="store_true", help="下载并解压后计算 zip 的 SHA256")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    splits = set(args.split)
    if "all" in splits:
        splits = set(URLS.keys())

    print(f"[info] COCO root = {root}")
    for split in ("train2017", "val2017", "annotations"):
        if split not in splits:
            continue
        url = URLS[split]
        zip_dst = root / Path(url).name
        _download_resumable(url, zip_dst)
        if args.check:
            digest = sha256_of(zip_dst)
            print(f"[sha256] {zip_dst.name}: {digest}")
        if not args.no_unzip:
            unzip_to(zip_dst, root)

    print("[done] COCO 下载流程结束。请回填 DATASET.md（sha256/下载时间）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())