from transformers import LayoutLMv2ForSequenceClassification


def load_model(pretrained_model_path, labels):
    model = LayoutLMv2ForSequenceClassification.from_pretrained(
        pretrained_model_path, num_labels=len(labels), local_files_only=True
    )
    
    return model
