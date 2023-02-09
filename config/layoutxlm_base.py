from .global_var import DOC_LABELS, DEVICE, SEED
PRETRAINED_TOKENIZER_PATH = "microsoft/layoutxlm-base"
LAYOUTXLM_CFG = {
    "data": {
        "custom": True,
        "path": "src/models/layoutxlm/load_data.py",
        "method": "load_data",
        "df_path": "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD_and_Samsung.csv",
        'pretrained_tokenizer_path': '/mnt/ssd500/hungbnt/DocumentClassification/weights/layoutxlm/tokenizer',
        'labels': DOC_LABELS,
        "image_shape": (3, 224, 224),
        "max_seq_len": 512,
        "batch_size": 8,
        'test_size': 0.2,
        "shuffle": True,
        "seed": SEED,
        "stratify": True,
        "num_workers": 16,
    },

    "model": {
        "custom": True,
        "path": "src/models/layoutxlm/load_model.py",
        "method": "load_model",
        "pretrained_model_path": '/mnt/ssd500/hungbnt/DocumentClassification/weights/layoutxlm/pretrained',
        "labels": DOC_LABELS
    },

    "optimizer": {
        "custom": True,
        "path": "src/models/layoutxlm/load_optimizer.py",
        "method": "load_optimizer",
        "lr": 5e-5,
    },

    "trainer": {
        "custom": True,
        "path": "src/models/layoutxlm/load_trainer.py",
        "method": "load_trainer",
        "labels": DOC_LABELS,
        "save_dir": '/mnt/ssd500/hungbnt/DocumentClassification/weights/layoutxlm/finetune',
        "n_epoches": 32,
        "device": DEVICE,
    },
}
