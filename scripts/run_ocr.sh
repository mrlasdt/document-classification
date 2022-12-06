export PYTHONPATH="$PYTHONPATH:/home/sds/thucpd/develop_OCR/TextDetectionApi/components/mmdetection"
export PYTHONPATH="$PYTHONPATH:/home/sds/datnt/mmocr"
export PYTHONWARNINGS="ignore"

while getopts i:o: flag
do
    case "${flag}" in
        i) img=${OPTARG};;
        o) out_dir=${OPTARG};;
    esac
done

python externals/ocr/api.py \
    --image  $img \
    --save_dir  "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/${out_dir}"\
    --det_cfg /home/sds/datnt/mmdetection/logs/textdet-fwd/yolox_s_8x8_300e_cocotext_1280.py \
    --det_ckpt /home/sds/datnt/mmdetection/logs/textdet-fwd/best_bbox_mAP_epoch_100.pth \
    --cls_cfg /home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py \
    --cls_ckpt /home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth
