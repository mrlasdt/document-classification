from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import re
from PIL import Image
from sty import fg, bg, ef, rs


class Trainer:
    def __init__(self, model, optimizer, labels, save_dir, n_epoches, device, task_prompt):
        self.model = model
        self.optimizer = optimizer
        self.labels = labels
        self.save_dir = save_dir
        self.n_epoches = n_epoches
        self.device = device
        self.task_prompt = task_prompt
        self.label2idx = {l: i for i, l in enumerate(labels)}

    def on_train_begin(self, train_dataloader, val_dataloader):
        train_tokenizer = train_dataloader.dataset.processor.tokenizer
        val_tokenizer = val_dataloader.dataset.processor.tokenizer
        assert len(train_tokenizer) == len(
            val_tokenizer), "Tokenizer of train and val dataset must have the same length, check the labels used to construct them"
        self.model.decoder.resize_token_embeddings(len(train_tokenizer))
        self.model.config.pad_token_id = train_tokenizer.pad_token_id
        self.model.config.decoder_start_token_id = train_tokenizer.convert_tokens_to_ids([self.task_prompt])[0]
        # sanity check
        processor = train_dataloader.dataset.processor
        assert processor.decode([self.model.config.pad_token_id]) == "<pad>", "wrokng pad token"
        assert processor.decode([self.model.config.decoder_start_token_id]
                                ) == self.task_prompt, "wrong task_prompt token"

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            # forward pass
            batch = {k: v.to(self.device) for k, v in batch.items()}
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]
            outputs = self.model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            running_loss += loss.item()
            # backward pass to get the gradients
            loss.backward()

            # update
            self.optimizer.step()
            self.optimizer.zero_grad()
        print("Loss:", running_loss / len(dataloader))

    def val_one_epoch(self, dataloader, best_acc):
        self.model.eval()
        dataset = dataloader.dataset
        processor = dataset.processor
        preds = []
        gts = []
        for i, d in tqdm(enumerate(dataset), total=len(dataset)):
            # prepare encoder inputs
            pixel_values = d["pixel_values"].to(self.device)
            # prepare decoder inputs
            decoder_input_ids = processor.tokenizer(
                self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(self.device)

            # autoregressively generate sequence
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            # turn into JSON
            seq = processor.batch_decode(outputs.sequences)[0]
            pred = dataset.token_sequence2label(seq)
            gt = dataset.token_sequence2label(d["labels"])
            preds.append(pred)
            gts.append(gt)
            if best_acc > 0.96 and pred != gt:  # for debug only #TODO: remove this
                print(dataset.df['img_path'].iloc[i])

        print(classification_report(gts, preds))
        return accuracy_score(gts, preds)

    def update_metric_and_save_model(self, best_acc, acc):
        if acc > best_acc:
            print(f"{fg.green}Accuracy updated from {best_acc} to {acc}{fg.rs}")
            best_acc = acc
            print("save new best model")
            self.model.save_pretrained(self.save_dir)
        print(f"{fg.blue} Current best accuracy: {best_acc}{fg.rs}")
        return best_acc

    def train(self, train_dataloader, val_dataloader):
        self.on_train_begin(train_dataloader, val_dataloader)
        best_acc = 0.0
        self.model.to(self.device)
        for epoch in range(self.n_epoches):
            print("Epoch:", epoch)
            self.train_one_epoch(train_dataloader)
            acc = self.val_one_epoch(val_dataloader, best_acc)
            best_acc = self.update_metric_and_save_model(best_acc, acc)


def load_trainer(model, optimizer, labels, save_dir, n_epoches, device, task_prompt):
    return Trainer(model, optimizer, labels, save_dir, n_epoches, device, task_prompt)
