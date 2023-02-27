export TOKENIZERS_PARALLELISM="true"
CUDA_VISIBLE_DEVICES=1 /home/sds/miniconda3/envs/hungbnt_kie_no_mmocr/bin/python main.py --cfg lilt | tee log/lilt_train.log