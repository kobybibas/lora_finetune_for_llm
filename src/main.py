import time
from datetime import datetime

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEBUG = False

MAX_EPOCHS = 5
MILESTONE = [4]
LR = 1e-3
BATCH_SIZE = 16
NUM_DATALOADER_WORKERS = 9

FINE_TUNE_TYPE = "lora"  # "lora", "entire_model"
assert FINE_TUNE_TYPE in ("lora", "entire_model")


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


def apply_lora_to_bert(model, rank=8, alpha=16, target_layers=None):
    """Apply LoRA to specific BERT attention layers"""
    if target_layers is None:
        target_layers = ["query", "value"]  # Q+V as discussed

    # Apply to all transformer layers
    for layer_idx in range(len(model.bert.encoder.layer)):
        layer = model.bert.encoder.layer[layer_idx]

        if "query" in target_layers:
            original_query = layer.attention.self.query
            layer.attention.self.query = LoRALinear(
                original_query, rank=rank, alpha=alpha
            )

        if "value" in target_layers:
            original_value = layer.attention.self.value
            layer.attention.self.value = LoRALinear(
                original_value, rank=rank, alpha=alpha
            )

        if "key" in target_layers:
            original_key = layer.attention.self.key
            layer.attention.self.key = LoRALinear(original_key, rank=rank, alpha=alpha)

        if "output" in target_layers:
            original_output = layer.attention.output.dense
            layer.attention.output.dense = LoRALinear(
                original_output, rank=rank, alpha=alpha
            )

    return model


class LitFinetuneModule(L.LightningModule):
    def __init__(self, model, learning_rate=LR, milestones=MILESTONE, warmup_steps=500):
        super().__init__()

        # Hyper-params
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        # Metrics
        self.compute_loss = nn.CrossEntropyLoss()
        self.compute_accuracy = Accuracy(task="multiclass", num_classes=2)
        self.training_step_loss_outputs = []
        self.validation_step_loss_outputs = []
        self.training_step_accuracy_outputs = []
        self.validation_step_accuracy_outputs = []

    def print_mps_memory(self):
        allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
        total = torch.mps.driver_allocated_memory() / 1024**3  # GB
        max_mem = torch.mps.recommended_max_memory() / 1024**3  # GB

        print(
            f"MPS Memory - Allocated: {allocated:.2f}GB, Total: {total:.2f}GB, Max: {max_mem:.2f}GB. Utilization: {(allocated / total) * 100:.1f}%"
        )

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

        if (batch_idx % 350) == 5:
            self.print_mps_memory()
            torch.mps.empty_cache()  # Clear unused cache
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

        lr = -1
        if len(self.trainer.optimizers) > 0:
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
    logger = CSVLogger(
        "logs", name=datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + FINE_TUNE_TYPE
    )

    print("-- Loading base model")
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if FINE_TUNE_TYPE == "lora":
        # Freeze all parameters except classifier and LoRA layers
        for param in model.parameters():
            param.requires_grad = False
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True
        print(model)
        print("apply_lora_to_bert")
        model = apply_lora_to_bert(model)
        print(model)
    elif FINE_TUNE_TYPE == "entire_model":
        # Unfreeze all parameters for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        print("Training entire model (no LoRA)")
        print(model)
    else:
        raise ValueError(f"Unknown FINE_TUNE_TYPE: {FINE_TUNE_TYPE}")

    lit_model = LitFinetuneModule(model=model)

    print("-- Loading dataset")
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"].shuffle().select(range(5000))
    val_dataset = tokenized_datasets["test"].shuffle().select(range(5000))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_DATALOADER_WORKERS,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_DATALOADER_WORKERS,
        persistent_workers=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = L.Trainer(
        max_epochs=1 if DEBUG else MAX_EPOCHS,
        accelerator=device,
        logger=logger,
        callbacks=[DeviceStatsMonitor()],
        limit_train_batches=10 if DEBUG else None,
        limit_val_batches=10 if DEBUG else None,
        limit_test_batches=10 if DEBUG else None,
        num_sanity_val_steps=0,
    )
    print("-- Training start")
    start_time = time.time()
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print(f"-- Training finish in {time.time() - start_time:.2f}")


if __name__ == "__main__":
    main()
