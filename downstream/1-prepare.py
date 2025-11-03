#!/usr/bin/env python3
import os
import glob
import gzip
import csv

def load_chromosomes(ref_dir):
    """
    Read all chr*.txt under ref_dir and return a dict: chrom -> sequence (str).
    Assumes filenames like chr1.txt ... chrY.txt, each containing plain base sequences (no header), possibly spanning multiple lines.
    """
    chrom_seqs = {}
    chr_names = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']
    for cname in chr_names:
        fn = os.path.join(ref_dir, cname, f"{cname}.txt")
        if not os.path.exists(fn):
            print(f"[!] Warning: {fn} not found, skipping.")
            continue
        with open(fn) as f:
            seq = f.readline().rstrip('\n')
        chrom_seqs[cname] = seq.upper()  # convert to uppercase
        print(f"[+] Loaded {cname}, length = {len(seq)}")

    return chrom_seqs

def get_window_size(name):
    """
    Return the extraction half-window size based on the dataset name.
    ATAC and CTCF: ±250bp (total length 500bp)
    promoter and enhancer: ±200bp (total length 400bp)
    Names starting with H3K27: ±500bp (total length 1000bp)
    """
    if name.startswith('H3K27'):
        return 500
    elif name.startswith('ATAC') or name.startswith('CTCF'):  # ATAC and CTCF
        return 250
    elif name.startswith('enhancer'):
        return 200
    elif name.startswith('promoter'):
        return 200
    else:
        raise ValueError(f"Unknown data type: {name}")

def extract_positive_to_csv(chrom_seqs, bed_dir):
    """
    Iterate over bed_dir/*.bed.gz, extract positive samples, and save to CSVs of the same base name.
    Extract sequences of different lengths by data type:
    - ATAC and CTCF: center ±100bp (total length 201bp)
    - Names starting with H3K27: center ±500bp (total length 1001bp)

    Output CSV columns:
    - chrom: chromosome name
    - sequence: extracted DNA sequence
    - label: label (1 indicates positive sample)
    - gc: count of G and C bases in the sequence
    """
    chr_names = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']

    for bed_path in sorted(glob.glob(os.path.join(bed_dir, '*.bed.gz'))):
        name = os.path.basename(bed_path).replace('.bed.gz', '')
        out_csv = os.path.join(bed_dir, "csvf", "pos", f"{name}_pos.csv")
        os.makedirs(os.path.join(bed_dir, "csvf", "pos"), exist_ok=True)
        window_size = get_window_size(bed_dir.split('/')[-1])
        print(f"[+] Processing {name} (window size: ±{window_size}bp) ...")
        
        total_in = 0
        total_out = 0
        with gzip.open(bed_path, 'rt') as bedf, open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['chrom', 'sequence', 'label', 'gc'])  # add gc column
            for line in bedf:
                if line.startswith('#') or not line.strip():
                    continue
                chrom, start, end = line.strip().split()[:3]
                if chrom not in chr_names:
                    continue

                # compute center position
                s = int(start)
                e = int(end)
                center = (s + e) // 2

                # extract sequence around the center
                seq_start = center - window_size
                seq_end = center + window_size

                # boundary checks
                if seq_start < s or seq_end > e:
                    continue

                seq = chrom_seqs[chrom][seq_start:seq_end]
                if len(seq) != window_size * 2:
                    continue
                total_in += 1
                if 'N' in seq:
                    continue

                # compute GC content
                gc_count = seq.count('G') + seq.count('C')
                writer.writerow([chrom, seq, 1, gc_count])  # add gc_count
                total_out += 1
                
        print(f"    → {name}: {total_out}/{total_in} kept (window size: ±{window_size}bp)")

    print(f"[✓] Finished extracting all files.")


def extract_negative_to_csv(chrom_seqs, bed_dir):
    """
    Iterate over bed_dir/*.bed.gz, extract negative samples, and save to CSVs of the same base name.
    Extract sequences of different lengths by data type:
    - ATAC and CTCF: center ±100bp (total length 201bp)
    - Names starting with H3K27: center ±500bp (total length 1001bp)

    Output CSV columns:
    - chrom: chromosome name
    - sequence: extracted DNA sequence
    - label: label (0 indicates negative sample)
    - gc: count of G and C bases in the sequence
    """
    chr_names = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']

    # get data type (ATAC/CTCF/H3K27ac, etc.)
    data_type = os.path.basename(os.path.dirname(bed_dir))
    # print(data_type)
    # raise Exception("stop")

    # create output directory
    out_dir = os.path.join(os.path.dirname(os.path.dirname(bed_dir)), data_type, "csvf", "neg")
    os.makedirs(out_dir, exist_ok=True)

    for bed_path in sorted(glob.glob(os.path.join(bed_dir, '*.bed.gz'))):
        name = os.path.basename(bed_path).replace('_neg.bed.gz', '')
        out_csv = os.path.join(out_dir, f"{name}_neg.csv")  # adjust output file name format
        # print(out_csv)
        # raise Exception("stop")
        window_size = get_window_size(data_type)
        print(f"[+] Processing {name} (window size: ±{window_size}bp) ...")
        
        total_in = 0
        total_out = 0
        with gzip.open(bed_path, 'rt') as bedf, open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['chrom', 'sequence', 'label', 'gc'])
            for line in bedf:
                if line.startswith('#') or not line.strip():
                    continue
                chrom, start, end = line.strip().split()[:3]
                if chrom not in chr_names:
                    continue

                # compute center position
                s = int(start)
                e = int(end)
                center = (s + e) // 2

                # extract sequence around the center
                seq_start = center - window_size
                seq_end = center + window_size

                # boundary checks
                # if seq_start < 0 or seq_end > len(chrom_seqs[chrom]):
                #     continue
                if seq_start < s or seq_end > e:
                    continue

                seq = chrom_seqs[chrom][seq_start:seq_end]
                if len(seq) != window_size * 2:
                    continue
                total_in += 1
                if 'N' in seq:
                    continue

                # compute GC content
                gc_count = seq.count('G') + seq.count('C')
                writer.writerow([chrom, seq, 0, gc_count])  # set label to 0 (negative)
                total_out += 1
                
        print(f"    → {name}: {total_out}/{total_in} kept (window size: ±{window_size}bp)")

    print(f"[✓] Finished extracting all negative samples.")


if __name__ == '__main__':
    chrom_seqs = load_chromosomes("/home/weicai/project/backup/pigmutbert/pig_mutbert/data/ref/raw_chrs")
    data_list = ["ATAC", "CTCF", "enhancer","H3K27ac", "H3K27me1", "H3K27me3", "promoter"]
    for data in data_list:
        # Extract positive/negative samples from xx.bed.gz files
        # Negative samples require running the sh script to generate via bedtools
        # Note: positive and negative samples must be processed separately
        # bed_dir = f"/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4/raw_files/{data}"
        bed_dir = f"/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4/raw_files/{data}/negbgz"
        # extract_positive_to_csv(chrom_seqs, bed_dir)
        extract_negative_to_csv(chrom_seqs, bed_dir)
