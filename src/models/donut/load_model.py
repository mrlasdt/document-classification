
# %%
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig


def load_model(pretrained_model_path, image_size, max_seq_len):
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_path)
    if model.config.encoder.image_size != image_size and model.config.decoder.max_length != max_seq_len:
        raise ValueError("[INFO]: Invalid config with the saved loading model path")
    return model


# %%
if __name__ == "__main__":
    # %% save model to local for the first time
    from pathlib import Path  # add parent path to run debugger
    import sys
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[2].as_posix())
    from config import config as cfg
    config = VisionEncoderDecoderConfig.from_pretrained(cfg.DONUT_DEFAULT_PRETRAINED_MODEL)
    config.encoder.image_size = cfg.DONUT_IMG_SIZE  # (height, width)
    config.decoder.max_length = cfg.DONUT_MAX_SEQ_LEN
    model = VisionEncoderDecoderModel.from_pretrained(cfg.DONUT_DEFAULT_PRETRAINED_MODEL, config=config)
    model.save_pretrained(cfg.DONUT_CFG['model']['pretrained_model_path'])
    print("Saved default model")

    # %%
    model_test = VisionEncoderDecoderModel.from_pretrained(cfg.DONUT_CFG['model']['pretrained_model_path'])
    assert model_test.config.encoder.image_size == list(cfg.DONUT_IMG_SIZE)
    assert model_test.config.decoder.max_length == cfg.DONUT_MAX_SEQ_LEN
    print("Test passed")
    # %%
    # model_test.config.encoder.image_size, cfg.DONUT_IMG_SIZE

    # %%
    # model_test.config.decoder.max_length, cfg.DONUT_MAX_SEQ_LEN
# %%
