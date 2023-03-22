from .global_var import DOC_LABELS, DEVICE, SEED
PRETRAINED_TOKENIZER_PATH = "vinai/phobert-base"
LILT_CFG = {
    "data": {
        "custom": True,
        "path": "src/models/lilt/load_data.py",
        "method": "load_data",
        "df_path": "data/FWD_and_Samsung.csv",
        'pretrained_tokenizer_path': 'weights/lilt/phobert_tokebizer',
        'labels': DOC_LABELS,
        "max_seq_len": 256,
        "batch_size": 8,
        'test_size': 0.2,
        "shuffle": True,
        "seed": SEED,
        "stratify": True,
        "num_workers": 16,
    },

    "model": {
        "custom": True,
        "path": "src/models/lilt/load_model.py",
        "method": "load_model",
        "pretrained_model_path": 'weights/lilt/pretrained',
        "labels": DOC_LABELS
    },

    "optimizer": {
        "custom": True,
        "path": "src/models/lilt/load_optimizer.py",
        "method": "load_optimizer",
        "lr": 5e-5,
    },

    "trainer": {
        "custom": True,
        "path": "src/models/lilt/load_trainer_huggingface.py",
        "method": "load_trainer",
        "labels": DOC_LABELS,
        "save_dir": 'weights/lilt/finetune',
        # "n_epoches": 32,
        # "device": DEVICE,
    },
}
