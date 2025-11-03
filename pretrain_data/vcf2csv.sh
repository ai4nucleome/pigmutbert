input_file="/path/to/pig_40_samples.vcf.gz"

output_file="/path/to/raw_trans.csv"

if [ -f "$input_file" ]; then
    echo "Processing $input_file -> $output_file"
    bcftools annotate --remove '^INFO/AN,INFO/AC' "$input_file" | \
    pv | \
    bcftools filter -e 'INFO/AC<=0' | \
    bcftools filter -e 'INFO/AN<=0' | \
    bcftools filter -i 'INFO/AC>0' | \
    bcftools view -v snps | \
    bcftools norm -m +any | \
    bcftools query -f '%CHROM,%POS,%REF,%ALT,%AC,%AN\n' -o "$output_file"

    # echo "finished"
else
    echo "File $input_file not found, skipping..."
fi