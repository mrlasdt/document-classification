#bash scripts/run_ocr.sh -i data/FWD/33forms -o results/ocr/FWD/33forms

export PYTHONWARNINGS="ignore"

while getopts i:o:d: flag
do
    case "${flag}" in
        i) img_dir=${OPTARG};;
        t) ocr_dir=${OPTARG};;
        o) out_file=${OPTARG};;
    esac
done

python src/tools/create_dataframe.py \
    --img_dir $img_dir \
    --ocr_dir $ocr_dir \
    --out_file   $out_file \
