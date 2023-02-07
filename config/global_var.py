# GLOBAL VARS
DEVICE = "cuda:1"
# DEVICE = "cpu"  # for debugging https://stackoverflow.com/questions/51691563/cuda-runtime-error-59-device-side-assert-triggered
# KIE_LABELS = ['gen', 'nk', 'nv', 'dobk', 'dobv', 'other']
IGNORE_LABEL = 'OTHERS'
# LABELS = ['POS01', 'POS02', 'POS03', 'POS04', 'POS05', 'POS06', 'POS08']
DOC_LABELS = ['POS01', 'POS02', 'POS03', 'POS04', 'POS05', 'POS06', 'POS08', "DCYCBH",
              "GUQ1", "GUQ2", "QLBH", "QLBHYT", "TDDG", "TKSK", "TTTK", "XNDY", "YCBH"]
SEED = 42
