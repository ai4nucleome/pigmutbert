#!/bin/bash
set -euo pipefail

# === Parameter settings ===
WORK_DIR="/home/weicai/project/backup/pigmutbert/down-stream/pig_dstreamv4"
GENOME_FILE="${WORK_DIR}/genome_file.txt"

# === Iterate over each data type directory ===, raw_files/*/
for data_dir in ${WORK_DIR}/raw_files/CTCFv2/; do
    data_name=$(basename ${data_dir})
    echo "处理目录: ${data_name}..."
    
    # Create output directories
    new_range_dir="${data_dir}/new_range"
    negbgz_dir="${data_dir}/negbgz"
    mkdir -p "${new_range_dir}" "${negbgz_dir}"
    
    # Process each bed file
    for bed_file in ${data_dir}/*.bed.gz; do
        base_name=$(basename ${bed_file} .bed.gz)
        echo "  处理文件: ${base_name}..."
        
        # Get the number of corresponding positive samples
        pos_csv="${data_dir}/csvf/pos/${base_name}_pos.csv"
        if [[ ! -f "${pos_csv}" ]]; then
            echo "  [!] 警告: ${pos_csv} 不存在，跳过"
            continue
        fi
        
        # Calculate the number of positive samples (subtract header row)
        pos_count=$(($(wc -l < "${pos_csv}") - 1))
        NEG_COUNT=$((pos_count * 10))
        MIN_NEG_COUNT=$((NEG_COUNT * 80 / 100))  # 80% of the target amount
        
        # === Step 1: generate new bed files (length varies by data type) ===
        echo "  [1] 生成新的bed文件..."
        
        # Set different lengths based on data type
        if [[ ${data_name} == "ATAC" || ${data_name} == "CTCFv2" ]]; then
            window_size=500
        elif [[ ${data_name} == H3K27* ]]; then
            window_size=1000
        elif [[ ${data_name} == "enhancer" || ${data_name} == "promoter" ]]; then
            window_size=400
        else
            echo "  [!] 警告: 未知的数据类型 ${data_name}，跳过"
            continue
        fi
        
        # Generate bed lines repeated 10x, set end value based on data type
        zcat "${bed_file}" | awk -v ws=${window_size} 'BEGIN{OFS="\t"} {end=$2+ws; for(i=1;i<=10;i++) print $1, $2, end}' \
        > "${new_range_dir}/${base_name}_newr.bed"
        
        # === Step 2: generate negative samples using bedtools shuffle ===
        echo "  [2] 生成负样本..."
        
        # First attempt: strict conditions
        bedtools shuffle -i "${new_range_dir}/${base_name}_newr.bed" \
                        -g "${GENOME_FILE}" \
                        -excl "${bed_file}" \
                        -chrom \
                        -noOverlapping \
                        -seed 42 \
                        -maxTries 10000 | \
            gzip -c > "${negbgz_dir}/${base_name}_neg.bed.gz"
            
        # Check the number of generated negative samples
        neg_count=$(zcat "${negbgz_dir}/${base_name}_neg.bed.gz" | wc -l)
        
        # If insufficient, regenerate with relaxed conditions
        if [[ ${neg_count} -lt ${MIN_NEG_COUNT} ]]; then
            echo "  [!] 警告: 负样本数量不足 (${neg_count}/${NEG_COUNT})，使用宽松条件重新生成..."
            bedtools shuffle -i "${new_range_dir}/${base_name}_newr.bed" \
                            -g "${GENOME_FILE}" \
                            -excl "${bed_file}" \
                            -noOverlapping \
                            -seed 42 \
                            -maxTries 10000 | \
                gzip -c > "${negbgz_dir}/${base_name}_neg.bed.gz"
                
            # Check the count again
            neg_count=$(zcat "${negbgz_dir}/${base_name}_neg.bed.gz" | wc -l)
            echo "  → 重新生成后数量: ${neg_count}/${NEG_COUNT}"
        fi
        
        echo "  ✅ 完成: 生成 ${neg_count} 个长度为 ${window_size}bp 的负样本"
    done
    
    # Clean up temporary directory
    rm -rf "${new_range_dir}"
done

echo "✅ 所有处理完成！"
