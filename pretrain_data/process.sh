for chr in {1..18} X Y; do
    out_dir="data/ref/raw_chrs/chr${chr}"
    mkdir -p "$out_dir"
    
    seqkit grep -n -r -p "chromosome:Sscrofa11.1:${chr}:" /path/to/Sus_scrofa.Sscrofa11.1.dna.toplevel.fa \
        | seqkit seq -u -g \
        | seqkit seq -s \
        > "${out_dir}/chr${chr}.txt"
done


