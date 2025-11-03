import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def process_merge_files():
    # Set working directory
    work_dir = "/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"
    data_types = ["ATAC", "CTCF", "CTCFv2", "enhancer","H3K27ac", "H3K27me1", "H3K27me3", "promoter"]
    
    # Iterate over each data type directory
    for dt in data_types:
        data_dir = os.path.join(work_dir, "raw_files", dt)
        if not os.path.exists(data_dir):
            continue
            
        print(f"[+] 处理数据类型: {dt}")
        merge_dir = os.path.join(data_dir, "csvf", "merge")
        split_dir = os.path.join(data_dir, "csvf", "split")
        os.makedirs(split_dir, exist_ok=True)
        
        # Iterate all csv files under merge directory
        for csv_file in glob.glob(os.path.join(merge_dir, "*.csv")):
            base_name = os.path.basename(csv_file).replace("_merge.csv", "")
            print(f"  [+] 处理文件: {base_name}")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # If sample count > 31250, randomly downsample to 31250
            max_samples = 31250
            if len(df) > max_samples:
                print(f"    [!] 样本数过多（{len(df)}），随机采样至{max_samples}")
                df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
            # First split dataset
            train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
            dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            # Create output directory for each subset
            output_dir = os.path.join(split_dir, base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each subset
            for name, sub_df in [
                ("train", train_df),
                ("dev", dev_df),
                ("test", test_df)
            ]:
                # Merge positive and negative samples
                sequences = pd.concat([sub_df['pos_seq'], sub_df['neg_seq']], ignore_index=True)
                labels = pd.concat([sub_df['pos_label'], sub_df['neg_label']], ignore_index=True)
                
                # Create a new DataFrame and shuffle
                result_df = pd.DataFrame({
                    'sequence': sequences,
                    'label': labels
                }).sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Save result
                output_file = os.path.join(output_dir, f"{name}.csv")
                result_df.to_csv(output_file, index=False)
                print(f"    → 保存{name}数据集: {len(result_df)} 个样本")
    
    print("[✓] 所有数据集处理完成！")

if __name__ == "__main__":
    process_merge_files()

