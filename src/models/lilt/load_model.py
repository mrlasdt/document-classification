from transformers import LiltForSequenceClassification


def load_model(pretrained_model_path, labels):
    model = LiltForSequenceClassification.from_pretrained(
        pretrained_model_path, num_labels=len(labels), local_files_only=True
    )

    return model
