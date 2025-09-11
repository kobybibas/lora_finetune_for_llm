from datetime import datetime

import lightning as L
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEBUG = True
MAX_EPOCHS = 20
MILESTONE = [10, 15]
LR = 1e-3
NUM_DATALOADER_WORKERS = 9


class LoRALayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int = 4, alpha: int = 1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.zeros((in_features, rank)))
        self.B = nn.Parameter(torch.zeros((rank, out_features)))

        torch.nn.init.xavier_uniform_(self.A)
        torch.nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        # x: (batch, in_features)
        z = torch.matmul(x, self.A)  # (batch, rank)
        z = torch.matmul(z, self.B)  # (batch, out_features)
        return (self.alpha / self.rank) * z  # Normalized scaling


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: int = 16):
        super().__init__()

        self.original_weight = original_linear.weight
        self.original_bias = original_linear.bias

        self.original_weight.requires_grad = False
        self.original_bias.requires_grad = False

        self.lora = LoRALayer(
            original_linear.in_features, original_linear.out_features, rank, alpha
        )

    def forward(self, x):
        with torch.no_grad():  # Extra protection against training the original layer
            original_out = F.linear(x, self.original_weight, self.original_bias)
        lora_out = self.lora(x)
        return original_out + lora_out


class LoRAFinetuneModule(L.LightningModule):
    def __init__(self, model, learning_rate=LR, milestones=MILESTONE, warmup_steps=500):
        super().__init__()

        # Hyper-params
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

        self.model = model

        # Metrics
        self.compute_loss = nn.CrossEntropyLoss()
        self.compute_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.training_step_loss_outputs = []
        self.validation_step_loss_outputs = []
        self.training_step_accuracy_outputs = []
        self.validation_step_accuracy_outputs = []

    def _step(self, batch):
        # batch['labels'], batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
        output = self.model(**batch)
        labels = batch["labels"]
        logits = output["logits"]
        loss = output["loss"]
        accuracy = self.compute_accuracy(logits, labels)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy_value = self._step(batch)
        self.training_step_loss_outputs.append(loss)
        self.training_step_accuracy_outputs.append(accuracy_value)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy_value = self._step(batch)
        self.validation_step_loss_outputs.append(loss)
        self.validation_step_accuracy_outputs.append(accuracy_value)
        return loss

    def on_validation_epoch_end(self):
        train_loss_mean, train_acc_mean = -1, -1
        if len(self.training_step_loss_outputs) > 0:
            train_loss_mean = torch.stack(self.training_step_loss_outputs).mean()
            train_acc_mean = torch.stack(self.training_step_accuracy_outputs).mean()
        val_loss_mean = torch.stack(self.validation_step_loss_outputs).mean()
        val_acc_mean = torch.stack(self.validation_step_accuracy_outputs).mean()
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        print(
            f"[Epoch {self.trainer.current_epoch}] Validation [Loss Acc]=[{val_loss_mean:.2f} {val_acc_mean:.2f}] Training [Loss Acc]=[{train_loss_mean:.2f} {train_acc_mean:.2f}] lr={lr:.2e}"
        )
        self.log("val_loss", val_loss_mean, on_step=False, on_epoch=True)
        self.log("val_accuracy", val_acc_mean, on_step=False, on_epoch=True)
        self.log("train_loss", train_loss_mean, on_step=False, on_epoch=True)
        self.log("train_accuracy", train_acc_mean, on_step=False, on_epoch=True)
        self.log("lr", lr, on_step=False, on_epoch=True)
        self.training_step_loss_outputs.clear()
        self.validation_step_loss_outputs.clear()

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=0.01)

        scheduler = None
        if self.milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.milestones, gamma=0.1
            )
        return [optimizer], [scheduler]


def main():
    import os

    print(os.getcwd())
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger("logs", name=now)

    print("Loading base model")
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(model)
    lit_model = LoRAFinetuneModule(model=model)

    print("Loading dataset")
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle().select(range(5000))
    small_eval_dataset = tokenized_datasets["test"].shuffle().select(range(5000))

    train_dataloader = DataLoader(
        small_train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=NUM_DATALOADER_WORKERS,
    )
    eval_dataloader = DataLoader(
        small_eval_dataset, batch_size=8, num_workers=NUM_DATALOADER_WORKERS
    )

    # TODO: apply LORA module 

    print("Training start")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=1 if DEBUG else MAX_EPOCHS,
        accelerator=device,
        logger=logger,
        limit_train_batches=100 if DEBUG else None,
        limit_val_batches=10 if DEBUG else None,
        limit_test_batches=10 if DEBUG else None,
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
    print("Training finish")


if __name__ == "__main__":
    main()
