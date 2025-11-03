import pysam
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
import io

def csv_post_process():
    """
    0: add header of csv files
    """
    
    chromosomes = [f'chr{i}.csv' for i in range(1, 23)] + ['chrX.csv']
    root_path = "/path/to/seqkit_trans"
    for chrom in chromosomes:
        data_path = os.path.join(root_path, f"trans/trans.{chrom}")
        save_path = os.path.join(root_path, f"clean/clean.{chrom}")
        # load csv
        try:
            df = pd.read_csv(data_path, header=None)
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
            print("Skipping problematic lines...")

            lines = []
            with open(data_path, 'r') as file:
                for line in file:
                    if len(line.strip().split(',')) == 5:
                        lines.append(line)

            df = pd.read_csv(io.StringIO('\n'.join(lines)), header=None)

        # add Header
        df.columns = ['CHROM', 'POS', 'REF', 'ALT', 'ALT_P']

        # add 'REF_P'
        df['REF_P'] = 1 - df['ALT_P']

        # write new file
        df.to_csv(save_path, index=False)


def fa2npy():
    """
    1st STEP: extract sequence data from hg38.fa.gz, save as chr_name.npy
    """
    # TODO
    fa_path = "/path/to/hg38.fa.gz"
    fa_data = pysam.FastaFile(fa_path)
    chr_names = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    save_root = "seq_npy_data"  # TODO
    for cname in chr_names:
        save_path = os.path.join(save_root, cname)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        chr_seq = fa_data.fetch(cname)
        np.save(os.path.join(save_path, f"{cname}.npy"), {"seq": chr_seq})


def split_by_n():
    """
    2nd STEP: split sequence data by "N" from chr_name.npy, save as chr_name_part_i.npy
    """
    root_path = "/home/weicai/projectnvme/pig_mutbert/data/ref/raw_chrs"
    save_root = "/home/weicai/projectnvme/pig_mutbert/data/ref"  # reference
    chr_names = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']
    min_len = 510

    for cname in tqdm(chr_names):
        data_path = os.path.join(root_path, f"{cname}/{cname}.txt")
        # npy_data = np.load(data_path, allow_pickle=True).item()
        # sequence = npy_data["seq"].upper()
        with open(data_path, "r") as f:
            sequence = f.readline().rstrip('\n')

        save_path = os.path.join(save_root, "2_split_data", f"{cname}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        length = len(sequence)
        n = 0
        start = 0
        while start < length:
            if sequence[start] == "N":
                start += 1
                continue
            for end in range(start, length):
                if sequence[end] == "N":
                    break
            split_length = end - start
            if split_length >= min_len:
                n += 1
                split_seq = sequence[start: end]
                split_range = (start, end)
                save_data = {"sp_seq": split_seq, "sp_range": split_range}
                np.save(os.path.join(save_path, f"{cname}_part_{n}.npy"), save_data)
        
            start = end + 1


def convert_seq2id(seq, bp_dict):
    bp_ids = []
    for bp in seq:
        bp_id = bp_dict[bp]
        bp_ids.append(bp_id)
    # pt_bp_ids = torch.tensor(bp_ids, dtype=torch.int64)
    return bp_ids

def create_sm_matrix():
    """
    3rd STEP: map str to float number, create smooth matrix
              from chr_name_part_i.npy (str) and clean.chr_name.csv, save as chr_name_part_i.npy (float)
    """
    root_path = "/home/weicai/projectnvme/pig_mutbert/data/ref/2_split_data"
    save_root = "/home/weicai/projectnvme/pig_mutbert/data/ref"

    bp_dict = {
        "A": 5,
        "C": 6,
        "G": 7,
        "T": 8
    }
    vocab_size = 9

    chr_names = [f'chr{i}' for i in range(1, 19)] + ['chrX', 'chrY']

    for cname in tqdm(chr_names):

        # get splitted data
        sm_chr_path = os.path.join(root_path, cname)
        # mut_data = pd.read_csv(os.path.join(root_path, f"{cname}/clean.{cname}.csv"))

        sm_chr_npy_list = os.listdir(sm_chr_path)
        sm_chr_npy_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        n = 0
        for nf in sm_chr_npy_list:  # [chrxx_part_i.npy]
            part_nf_path = os.path.join(sm_chr_path, nf)
            part_nf_data = np.load(part_nf_path, allow_pickle=True).item()
            split_seq = part_nf_data["sp_seq"]      # str
            start, end = part_nf_data["sp_range"]   # list, [start, end), from 0

            # convert integer to one hot matrix
            bp_ids = convert_seq2id(split_seq, bp_dict)
            pt_bp_ids = torch.tensor(bp_ids, dtype=torch.int64)
            oh_input_ids = F.one_hot(pt_bp_ids, num_classes=vocab_size).float()

            # sp_mut_df = mut_data[(mut_data['POS'] >= start + 1) & (mut_data['POS'] <= end)] 
            # if not sp_mut_df.empty:
            #     adjusted_pos = sp_mut_df["POS"] - start - 1  # list

            #     pt_refs = convert_seq2id(sp_mut_df["REF"].to_list(), bp_dict)
            #     pt_alts = convert_seq2id(sp_mut_df["ALT"].to_list(), bp_dict)

            #     poss = torch.tensor(adjusted_pos.to_list(), dtype=torch.int64)
            #     refs = torch.tensor(pt_refs, dtype=torch.int64)
            #     alts = torch.tensor(pt_alts, dtype=torch.int64)
            #     ref_ps = torch.tensor(sp_mut_df["REF_P"].to_list(), dtype=torch.float32)
            #     alt_ps = torch.tensor(sp_mut_df["ALT_P"].to_list(), dtype=torch.float32)

            #     # modify, get smooth matrix
            #     oh_input_ids[poss, refs] = ref_ps
            #     oh_input_ids[poss, alts] = alt_ps
            
            oh_input_ids = oh_input_ids.numpy()
            n += 1

            save_path = os.path.join(save_root, "3_mtx_data", f"{cname}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            np.save(os.path.join(save_path, f"{cname}_part_{n}.npy"), oh_input_ids)


def cat_all_npy(vocab_size=9):
    """
    4th STEP: concatenate all the interval smooth matrix
              from chr_name_part_i.npy (float), save as train_data.npy and test_data.npy
    """
    root_path = "/home/weicai/projectnvme/pig_mutbert/data/ref/3_mtx_data"
    save_root = "/home/weicai/projectnvme/pig_mutbert/data/ref"
    # train_names = [f'chr{i}' for i in range(1, 18)] + ['chrX', 'chrY']  # train dataset
    train_names = ["chr18"]  # test dataset 
    # train: chr1-17,X,Y  ------  test: chr 18
    N_total = 0
    for name in train_names:
        chr_path = os.path.join(root_path, name)
        chr_npy_list = os.listdir(chr_path)  # chrxx_part_i.npy
        chr_npy_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for nf in chr_npy_list:
            chr_npy_path = os.path.join(chr_path, nf)
            cnpy = np.load(chr_npy_path, mmap_mode="r")
            N_total = N_total + cnpy.shape[0] + 1
    
    output_shape = (N_total, vocab_size)  # x 个 sep
    # TODO: if you create test_data.npy using chr22, please replace "train_data.npy" with "test_data.npy"
    merged_file = np.memmap(os.path.join(save_root, "test_data.npy"), dtype=np.float32, mode='w+', shape=output_shape)
    sep_data = np.array([0,0,1,0,0,0,0,0,0]).reshape(1, -1)  # [sep] one hot
    # tokenization, sep = 2

    # load and write data
    current_index = 0
    for name in train_names:

        chr_path = os.path.join(root_path, name)
        chr_npy_list = os.listdir(chr_path)  # chrxx_part_i.npy
        chr_npy_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for nf in chr_npy_list:
            chr_npy_path = os.path.join(chr_path, nf)
            data = np.load(chr_npy_path, mmap_mode="r")
            N_i = data.shape[0]

            merged_file[current_index:current_index + N_i] = data
            merged_file[current_index + N_i] = sep_data
            current_index = current_index + N_i + 1 # we have [sep] between 2 segments 

    print("current index:", current_index)
    print("N_total:", N_total)
    print("shape:", output_shape)

    # flush data
    merged_file.flush()
    del merged_file


if __name__ == "__main__":
    # csv_post_process()   # 0
    # fa2npy()             # 1
    # split_by_n()         # 2
    # create_sm_matrix()   # 3
    cat_all_npy()        # 4
    