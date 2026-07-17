"""Temporary script to compute correctly padded markdown tables.

Padding uses display width (east-asian wide chars count as 2) to match
markdownlint MD060, which measures alignment via the string-width package.
"""
import sys
import unicodedata


def display_width(s):
    """Display width of a string: F/W east-asian chars are 2 columns wide."""
    return sum(2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1 for ch in s)

# Table 1: 压榨参数
rows1 = [
    ['参数', '值', '理由'],
    ['`batch`', '512', 'RTX 5090 32GB，YOLO11n @416 约 18GB，充裕'],
    ['`workers`', '24', '25 vCPU 留 1 个给系统'],
    ['`imgsz`', '416', '比本地 320 高，小目标精度更好'],
    ['`cache`', 'ram', '13576 张图约 3GB，90GB RAM 充裕'],
    ['`amp`', 'True', '混合精度，省显存+提速'],
    ['`epochs`', '180', '完整训练'],
    ['`cos_lr`', 'True', 'cosine 学习率'],
    ['`patience`', '30', '30 epoch 无提升早停'],
]

# Table 2: 4 个变体
rows2 = [
    ['变体', 'YAML', '说明', '论文角色'],
    ['`baseline`', 'yolo11n_lead.yaml', 'YOLO11n 7类，无 LCA', 'RQ1 对照基准'],
    ['`lca_r16`', 'yolo11n_lca_neck_r16.yaml', '+LCA(Neck, r=16)', 'RQ1 主实验'],
    ['`lca_r8`', 'yolo11n_lca_neck_r8.yaml', '+LCA(Neck, r=8)', 'RQ2 reduction 消融'],
    ['`lca_r32`', 'yolo11n_lca_neck_r32.yaml', '+LCA(Neck, r=32)', 'RQ2 reduction 消融'],
]

# Table 3: 故障排除
rows3 = [
    ['问题', '解决'],
    ['`ModuleNotFoundError: lead_net`', '确认 `export PYTHONPATH=.`'],
    ['显存不足 OOM', '降 batch 到 256 或 128'],
    ['GPU 利用率低', '确认 cache=ram 生效，workers 调到 24'],
    ['数据集找不到', '检查 data/lead_subset/images/train 存在'],
    ['权重迁移失败', '检查 yolo11n.pt 在项目根目录（首次运行自动下载）'],
]


def build_table(rows):
    cols = len(rows[0])
    # compute max display width per column
    widths = [0] * cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))
    lines = []
    for idx, row in enumerate(rows):
        cells = []
        for i, cell in enumerate(row):
            pad = widths[i] - display_width(cell)
            cells.append(' ' + cell + ' ' * pad + ' ')
        lines.append('|' + '|'.join(cells) + '|')
        if idx == 0:
            seps = []
            for w in widths:
                seps.append('-' * (w + 2))
            lines.append('|' + '|'.join(seps) + '|')
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--write' in sys.argv:
        doc = 'docs/CLOUD_TRAIN_GUIDE.md'
        with open(doc, encoding='utf-8') as f:
            lines = f.read().split('\n')
        # 1-based inclusive line ranges of the three tables in the doc
        for start, end, rows in [(150, 156, rows3), (141, 146, rows2), (59, 68, rows1)]:
            lines[start - 1:end] = build_table(rows).split('\n')
        with open(doc, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines))
        print(f'updated {doc}')
    elif '--t1' in sys.argv:
        print(build_table(rows1))
    elif '--t2' in sys.argv:
        print(build_table(rows2))
    elif '--t3' in sys.argv:
        print(build_table(rows3))
    else:
        print('=== TABLE 1 ===')
        print(build_table(rows1))
        print()
        print('=== TABLE 2 ===')
        print(build_table(rows2))
        print()
        print('=== TABLE 3 ===')
        print(build_table(rows3))
