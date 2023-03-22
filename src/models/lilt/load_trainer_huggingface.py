# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb
import evaluate
from transformers import TrainingArguments, Trainer

metric = evaluate.load("seqeval")
import numpy as np
from seqeval.metrics import classification_report

return_entity_level_metrics = False
from functools import partial


def compute_metrics(p, idx2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    predictions = [[idx2label[pred]] for pred in predictions]
    labels = [[idx2label[l]] for l in labels]
    # print(predictions, labels)
    results = metric.compute(predictions=predictions, references=labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


training_args = TrainingArguments(output_dir="weights/lilt/finetune",
                                  num_train_epochs=32,
                                  learning_rate=5e-5,
                                  evaluation_strategy="steps",
                                  eval_steps=100,
                                  load_best_model_at_end=False,
                                  metric_for_best_model="f1")


class CustomTrainer(Trainer):

    def __init__(self, model, args, compute_metrics):
        self.train_dataloader = None
        self.eval_dataloader = None
        super().__init__(model=model, args=args, compute_metrics=compute_metrics)

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        return self.eval_dataloader

    def train(self, train_dataloader, eval_dataloder):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloder
        super().train()


# Initialize our Trainer
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )


def load_trainer(model, optimizer, labels, save_dir):
    idx2label = {id: label for id, label in enumerate(labels)}
    compute_metrics_with_labels = partial(compute_metrics, idx2label=idx2label)
    return CustomTrainer(model=model, args=training_args, compute_metrics=compute_metrics_with_labels)

# NOTE: huggingface trainer required different interface to our trainer -> uncessary
