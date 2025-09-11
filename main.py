import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
import lightning as L


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
        with torch.no_grad(): # Extra protection against training the original layer
            original_out = F.linear(x, self.original_weight, self.original_bias)
        lora_out = self.lora(x)
        return original_out + lora_out
    


class LoRAFinetuningModule(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4, milestones=None, warmup_steps=500):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()
        
        # Track baseline metrics on first validation
        self.baseline_logged = False
    
    def training_step(self, batch, batch_idx):
        # TODO
        # Return loss and log metrics
        pass
    
    def validation_step(self, batch, batch_idx):
        # TODO
        # Calculate metrics and return them
        pass
    
    def on_validation_epoch_end(self):
        # TODO
        # Log baseline vs current performance comparison
        # This is where you track improvement
        pass
    
    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=0.01)

        scheduler = None
        if self.milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, milestones=self.milestones, gamma=0.1)
        return [optimizer], [scheduler]



def main():
    print("Hello from lora-finetune-for-llm!")


    # Create test layer
    original = nn.Linear(512, 256)
    lora_layer = LoRALinear(original, rank=8, alpha=16)

    # Verify shapes and gradients
    x = torch.randn(32, 512)
    output = lora_layer(x)
    print(f"Output shape: {output.shape}")  # Should be [32, 256]

    # Check which parameters are trainable
    for name, param in lora_layer.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")


if __name__ == "__main__":
    main()
