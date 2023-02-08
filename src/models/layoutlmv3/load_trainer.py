import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sty import fg, bg, ef, rs
from terminaltables import AsciiTable


class Trainer:
    def __init__(self, model, optimizer, labels, save_dir, n_epoches, device):
        self.model = model
        self.optimizer = optimizer
        self.labels = labels
        self.save_dir = save_dir
        self.n_epoches = n_epoches
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        for batch in tqdm(dataloader):
            # forward pass
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch['labels']).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            self.optimizer.step()
            self.optimizer.zero_grad()

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / len(dataloader.dataset)
        print("Training accuracy:", accuracy.item())

    def val_one_epoch(self, dataloader):
        self.model.eval()
        correct = 0
        preds, truths = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(-1)

                preds.extend(predictions.detach().cpu().numpy().tolist())
                truths.extend(batch['labels'].detach().cpu().numpy().tolist())
                correct += (predictions == batch['labels']).float().sum()

        accuracy = 100 * correct / len(dataloader.dataset)
        p, r, f1, support = precision_recall_fscore_support(truths, preds)
        table_data = [["Class", "P", "R", "F1", "#samples"]]
        for c in range(len(self.labels)):
            table_data.append([self.labels[c], p[c], r[c], f1[c], support[c]])
        table = AsciiTable(table_data)
        print(table.table)
        print(
            "Validation accuracy:", accuracy.item(),
            "- #samples:", len(dataloader.dataset),
            "- #corrects:", correct
        )
        return accuracy

    def update_metric_and_save_model(self, best_acc, acc):
        if acc > best_acc:
            print(f"{fg.green}Accuracy updated from {best_acc} to {acc}{fg.rs}")
            best_acc = acc
            print("Save new best model")
            self.model.save_pretrained(self.save_dir)
        print(f"{fg.blue} Current best accuracy: {best_acc}{fg.rs}")
        return best_acc

    def train(self, train_dataloader, val_dataloader):
        r"""Train LayoutXLM model"""
        self.model = self.model.to(self.device)
        best_acc = 0.0
        for epoch in range(self.n_epoches):
            print("Epoch:", epoch)
            self.train_one_epoch(train_dataloader)
            acc = self.val_one_epoch(val_dataloader)
            best_acc = self.update_metric_and_save_model(best_acc, acc)


def load_trainer(model, optimizer, labels, save_dir, n_epoches, device):
    return Trainer(model, optimizer, labels, save_dir, n_epoches, device)
