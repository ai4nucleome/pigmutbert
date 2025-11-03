#!/bin/bash
set -euo pipefail

# One-click pipeline for downstream processing.
# Steps:
# 1) Run positive extraction using functions from 1-prepare.py
# 2) Run 2-prepare_neg.sh to generate 10x negatives (bed.gz)
# 3) Run negative extraction using functions from 1-prepare.py
# 4) Run 3-prepare_merge.py to match pos/neg
# 5) Run 4-merge.py to split into train/dev/test subsets

ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
DOWNSTREAM_DIR="${ROOT_DIR}/downstream"

# Keep paths consistent with existing scripts
CHR_DIR="/home/weicai/project/backup/pigmutbert/pig_mutbert/data/ref/raw_chrs"
WORK_DIR="/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"

echo "[Pipeline] Step 1/5: Extract positive samples (1-prepare.py functions)"
python3 - <<'PYEOF'
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DOWNSTREAM_DIR = os.path.join(ROOT_DIR, 'downstream')

# Keep the chromosome directory path aligned with existing usage
CHR_DIR = "/home/weicai/project/backup/pigmutbert/pig_mutbert/data/ref/raw_chrs"
WORK_DIR = "/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"

# Load functions from 1-prepare.py without changing the original file
prepare_py = os.path.join(DOWNSTREAM_DIR, '1-prepare.py')
ns = {}
with open(prepare_py, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, prepare_py, 'exec'), ns, ns)

load_chromosomes = ns['load_chromosomes']
extract_positive_to_csv = ns['extract_positive_to_csv']

chrom_seqs = load_chromosomes(CHR_DIR)

# Data types for positive extraction (aligned with 1-prepare.py)
data_list = ["ATAC", "CTCF", "enhancer", "H3K27ac", "H3K27me1", "H3K27me3", "promoter"]

for data in data_list:
    bed_dir = os.path.join(WORK_DIR, 'raw_files', data)
    if os.path.exists(bed_dir):
        extract_positive_to_csv(chrom_seqs, bed_dir)
    else:
        print(f"[!] Warning: {bed_dir} not found, skip positive extraction.")
PYEOF

echo "[Pipeline] Step 2/5: Generate negative bed.gz (2-prepare_neg.sh)"
bash "${DOWNSTREAM_DIR}/2-prepare_neg.sh"

echo "[Pipeline] Step 3/5: Extract negative samples (1-prepare.py functions)"
python3 - <<'PYEOF'
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DOWNSTREAM_DIR = os.path.join(ROOT_DIR, 'downstream')

CHR_DIR = "/home/weicai/project/backup/pigmutbert/pig_mutbert/data/ref/raw_chrs"
WORK_DIR = "/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"

prepare_py = os.path.join(DOWNSTREAM_DIR, '1-prepare.py')
ns = {}
with open(prepare_py, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, prepare_py, 'exec'), ns, ns)

load_chromosomes = ns['load_chromosomes']
extract_negative_to_csv = ns['extract_negative_to_csv']

chrom_seqs = load_chromosomes(CHR_DIR)

# Data types for negative extraction (aligned with 1-prepare.py)
data_list = ["ATAC", "CTCF", "enhancer", "H3K27ac", "H3K27me1", "H3K27me3", "promoter"]

for data in data_list:
    bed_dir = os.path.join(WORK_DIR, 'raw_files', data, 'negbgz')
    if os.path.exists(bed_dir):
        extract_negative_to_csv(chrom_seqs, bed_dir)
    else:
        print(f"[!] Warning: {bed_dir} not found, skip negative extraction.")
PYEOF

echo "[Pipeline] Step 4/5: Match positives and negatives (3-prepare_merge.py)"
python3 "${DOWNSTREAM_DIR}/3-prepare_merge.py"

echo "[Pipeline] Step 5/5: Split into train/dev/test (4-merge.py)"
python3 "${DOWNSTREAM_DIR}/4-merge.py"

echo "[Pipeline] Done."


