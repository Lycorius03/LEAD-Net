"""Fix markdown tables and fenced code block language in CLOUD_TRAIN_GUIDE.md."""
import re

md_path = r'f:\LEAD-Net\docs\CLOUD_TRAIN_GUIDE.md'

with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# ---- Fix 1: MD040 - add language to fenced code block ----
for i, line in enumerate(lines):
    if '### 3.1' in line:
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() == '```':
                lines[j] = '```text'
                break
        break

# ---- Fix 2: Tables ----

def is_separator_cell(cell):
    """Check if a cell is a separator (only dashes, optional leading/trailing spaces, optional colon)."""
    stripped = cell.strip()
    if not stripped:
        return False
    # Allow :---: style alignment markers
    cleaned = stripped.replace(':', '')
    return len(cleaned) > 0 and all(c == '-' for c in cleaned)

def is_separator_row(line):
    """Check if a table line is a separator row."""
    stripped = line.strip()
    if not stripped.startswith('|'):
        return False
    cells = stripped.split('|')
    # Remove empty first/last
    if cells and cells[0] == '':
        cells = cells[1:]
    if cells and cells[-1] == '':
        cells = cells[:-1]
    if not cells:
        return False
    return all(is_separator_cell(c) for c in cells)

def find_tables(lines):
    """Find all table regions. Returns list of (start, end) indices."""
    tables = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('|') and stripped.count('|') >= 2:
            start = i
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().count('|') >= 2:
                i += 1
            tables.append((start, i))
        else:
            i += 1
    return tables

def parse_table_rows(lines_slice):
    """Parse table rows, skipping separator rows. Returns (header, data_rows)."""
    data_rows = []
    for line in lines_slice:
        if is_separator_row(line):
            continue  # skip old separator rows
        cells = [c.strip() for c in line.strip().split('|')]
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        data_rows.append(cells)
    return data_rows

def build_table(rows):
    """Build properly aligned table lines from parsed rows."""
    if not rows:
        return []
    cols = len(rows[0])
    # Compute max text width per column
    widths = [0] * cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    result = []
    for idx, row in enumerate(rows):
        cells = []
        for i, cell in enumerate(row):
            cells.append(' ' + cell.ljust(widths[i]) + ' ')
        result.append('|' + '|'.join(cells) + '|')
        if idx == 0:
            # separator row after header
            seps = []
            for w in widths:
                seps.append('-' * (w + 2))
            result.append('|' + '|'.join(seps) + '|')
    return result

tables = find_tables(lines)
print(f'Found {len(tables)} table regions')

# Process from bottom to top to preserve indices
for start, end in reversed(tables):
    table_lines = lines[start:end]
    rows = parse_table_rows(table_lines)
    new_lines = build_table(rows)
    lines[start:end] = new_lines

# Write back
result = '\n'.join(lines)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(result)

print('Done. File updated.')
