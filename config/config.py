# GLOBAL VARS
DEVICE = "cuda:1"
# DEVICE = "cpu"
# DEVICE = "cpu"  # for debugging https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
# DEVICE = "cpu"
# KIE_LABELS = ['gen', 'nk', 'nv', 'dobk', 'dobv', 'other']
IGNORE_KIE_LABEL = 'others'
KIE_LABELS = ['id', 'name', 'dob', 'home', 'add', 'sex', 'nat', 'exp', 'eth', 'rel', 'date', 'org', IGNORE_KIE_LABEL]
SEED = 42

##########################################
BASE = {
    "data": {
        "custom": True,
        "path": "src/custom/load_data.py",
        "method": "load_data",
        "train_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/synthesis_for_train/",
        "val_path": "/home/sds/hoangmd/TokenClassification_copy/giaykhaisinh/SDV_Meddoc_BirthCert/",
        # "size": 320,
        "max_seq_len": 512,
        "batch_size": 8,
        # "workers": 10,
        'pretrained_processor': 'microsoft/layoutxlm-base',
        'kie_labels': KIE_LABELS,
        'device': DEVICE,
    },

    "model": {
        "custom": True,
        "path": "src/custom/load_model.py",
        "method": "load_model",
        "pretrained_model": 'microsoft/layoutxlm-base',
        'kie_labels': KIE_LABELS,
        'device': DEVICE,
    },

    "optimizer": {
        "custom": True,
        "path": "src/custom/load_optimizer.py",
        "method": "load_optimizer",
        "lr": 5e-6,
        "weight_decay": 0,  # default = 0
        "betas": (0.9, 0.999),  # beta1 in transformer, default = 0.9
    },

    "trainer": {
        "custom": True,
        "path": "src/custom/load_trainer.py",
        "method": "load_trainer",
        "kie_labels": KIE_LABELS,
        "save_dir": 'weights',
        "n_epoches": 100,
    },
}
