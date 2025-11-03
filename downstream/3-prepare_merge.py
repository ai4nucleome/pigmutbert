#!/usr/bin/env python3
import os
import glob
import pandas as pd
import csv
from collections import defaultdict, deque
from tqdm import tqdm

# Global chromosome order, used for cross-chromosome attempts
CHROM_ORDER = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']

def load_and_sort_samples(csv_path):
    """Load chrom, gc, sequence, label and sort by chrom and gc."""
    chrom_cat = pd.CategoricalDtype(categories=CHROM_ORDER, ordered=True)
    df = pd.read_csv(
        csv_path,
        usecols=['chrom', 'gc', 'sequence', 'label']
    )
    df['chrom'] = df['chrom'].astype(chrom_cat)
    return df.sort_values(['chrom','gc'], ascending=[True, False])\
             .reset_index(drop=True)

def build_neg_map(neg_df):
    """
    Build a mapping (chrom, gc) -> deque([row_idx,...]) for neg_df,
    supporting amortized O(1) same-chromosome and cross-chromosome matching.
    """
    neg_map = defaultdict(deque)
    for idx, row in neg_df.iterrows():
        neg_map[(row.chrom, row.gc)].append(idx)
    return neg_map

def match_same_chrom(pos, neg_map):
    """
    Try same-chromosome first:
      1) exact match (chrom, gc)
      2) ±1/±2 GC
    """
    chrom, gc = pos.chrom, pos.gc
    # 1) 完全匹配
    if neg_map[(chrom, gc)]:
        return neg_map[(chrom, gc)].popleft()
    # 2) ±1, ±2 匹配
    for delta in (-1, +1, -2, +2):
        key = (chrom, gc + delta)
        if neg_map[key]:
            return neg_map[key].popleft()
    return None

def match_cross_chrom(pos, neg_map):
    """
    Cross-chromosome matching: among other chromosomes, try GC difference ≤ 2,
    prioritizing GC deltas (0, ±1, ±2), then chromosome order.
    """
    gc = pos.gc
    for delta in (0, -1, +1):  # ± 2
        target_gc = gc + delta
        for chrom in CHROM_ORDER:
            if chrom == pos.chrom:
                continue
            key = (chrom, target_gc)
            if neg_map[key]:
                return neg_map[key].popleft()
    return None

def process_file_pair(pos_file, neg_file, out_file, match_log_fh=None):
    """
    Process a pos/neg file pair, write the merged file, and return match counts.
    """
    pos_df = load_and_sort_samples(pos_file)
    neg_df = load_and_sort_samples(neg_file)
    neg_map = build_neg_map(neg_df)

    matched_count = 0
    total_count   = len(pos_df)

    with open(out_file, 'w', newline='') as fo:
        writer = csv.DictWriter(fo, fieldnames=[
            'pos_chr','pos_seq','pos_gc','pos_label',
            'neg_chr','neg_seq','neg_gc','neg_label'
        ])
        writer.writeheader()

        for _, pos in tqdm(pos_df.iterrows(), total=total_count,
                          desc=f"处理 {os.path.basename(pos_file)}"):
            # 1) same-chromosome match (exact or ±2 GC)
            neg_idx = match_same_chrom(pos, neg_map)
            # 2) if not matched, try cross-chromosome ±2 GC
            if neg_idx is None:
                neg_idx = match_cross_chrom(pos, neg_map)

            # 3) if found, write the result
            if neg_idx is not None:
                neg = neg_df.loc[neg_idx]
                writer.writerow({
                    'pos_chr':   pos.chrom,   'pos_seq':   pos.sequence,
                    'pos_gc':    pos.gc,      'pos_label': pos.label,
                    'neg_chr':   neg.chrom,   'neg_seq':   neg.sequence,
                    'neg_gc':    neg.gc,      'neg_label': neg.label,
                })
                matched_count += 1

    # write matching log
    if match_log_fh:
        if matched_count == 0:
            match_log_fh.write(f"{os.path.basename(out_file)}: 没有匹配到任何样本对\n")
        else:
            match_log_fh.write(
                f"{os.path.basename(out_file)}: 成功匹配 {matched_count}/{total_count} 对样本\n"
            )

    return matched_count, total_count

def process_data_type(data_dir, match_file):
    data_name = os.path.basename(data_dir)
    print(f"[+] 处理数据类型: {data_name}")

    merge_dir = os.path.join(data_dir, "csvf", "merge")
    os.makedirs(merge_dir, exist_ok=True)

    with open(match_file, 'a') as log_fh:
        for pos_file in glob.glob(os.path.join(data_dir, "csvf", "pos", "*_pos.csv")):
            base = os.path.basename(pos_file).replace("_pos.csv","")
            neg_file = os.path.join(data_dir, "csvf", "neg", f"{base}_neg.csv")
            out_file = os.path.join(merge_dir, f"{base}_merge.csv")

            if not os.path.exists(neg_file):
                print(f"  [!] 警告: 缺少负样本 {neg_file}")
                log_fh.write(f"{data_name}/{base}: 负样本文件不存在\n")
                continue

            print(f"  [+] 处理文件对: {base}")
            m, t = process_file_pair(pos_file, neg_file, out_file, match_log_fh=log_fh)
            print(f"    → 成功匹配 {m}/{t}")

def main():
    work_dir  = "/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"
    # types     = ["ATAC", "CTCF", "enhancer","H3K27ac", "H3K27me1", "H3K27me3", "promoter"]
    types     = ["CTCFv2"]
    match_txt = os.path.join(os.path.dirname(__file__), "match.txt")

    for dt in types:
        data_dir = os.path.join(work_dir, "raw_files", dt)
        if os.path.exists(data_dir):
            process_data_type(data_dir, match_txt)

    print("[✓] 所有数据处理完成！")

if __name__ == "__main__":
    main()
