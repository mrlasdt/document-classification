# GLOBAL VARS
DEVICE = "cuda:0"
# DEVICE = "cpu"  # for debugging https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
# KIE_LABELS = ['gen', 'nk', 'nv', 'dobk', 'dobv', 'other']
IGNORE_LABEL = 'OTHERS'
LABELS = ['POS01', 'POS02', 'POS03', 'POS04', 'POS05', 'POS06', 'POS08']
SEED = 42
##### DONUT CONFIG ##########################################
# re run load_model.py as main if any change in IMG_SIZE AND MAX_SEQ_LEN
# DONUT_IMG_SIZE = (1280, 960)  # (height, width) too large, have to run with batch_size <2
DONUT_IMG_SIZE = (640, 480)  # (height, width)
DONUT_MAX_SEQ_LEN = 8  # must be > 4
DONUT_DEFAULT_PRETRAINED_MODEL = "naver-clova-ix/donut-base"
DONUT_TASK_PROMPT = "<FWD>"
DONUT_CFG = {
    "data": {
        "custom": True,
        "path": "src/donut/load_donut_dataloader.py",
        "method": "load_data",
        "df_path": "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD.csv",
        'pretrained_processor_path': '/mnt/ssd500/hungbnt/DocumentClassification/weights/donut/pretrained/clova_donut_processor',
        "task_start_token": DONUT_TASK_PROMPT,
        "prompt_end_token": DONUT_TASK_PROMPT,
        'labels': LABELS,
        "image_size": DONUT_IMG_SIZE,
        "max_seq_len": DONUT_MAX_SEQ_LEN,
        "batch_size": 12,
        'test_size': 0.2,
        "shuffle": True,
        "seed": SEED,
        "stratify": True
    },

    "model": {
        "custom": True,
        "path": "src/donut/load_model.py",
        "method": "load_model",
        "pretrained_model_path": '/mnt/ssd500/hungbnt/DocumentClassification/weights/donut/pretrained/clova_donut_model',
        'image_size': DONUT_IMG_SIZE,
        'max_seq_len': DONUT_MAX_SEQ_LEN,
    },

    "optimizer": {
        "custom": True,
        "path": "src/donut/load_optimizer.py",
        "method": "load_optimizer",
        "lr": 1e-5,
    },

    "trainer": {
        "custom": True,
        "path": "src/custom/load_trainer.py",
        "method": "load_trainer",
        "labels": LABELS,
        "save_dir": '/mnt/ssd500/hungbnt/DocumentClassification/weights/donut',
        "n_epoches": 5,
        "device": DEVICE,
        "task_prompt": DONUT_TASK_PROMPT
    },
}
