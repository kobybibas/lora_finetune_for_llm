# LoRA Fine-tuning with PyTorch (Lightning)
Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) technique. 
Built as a learning exercise to understand the mathematical foundations and practical implementation of LoRA for adapting large language models to new datasets.

Model used: bert-base-uncased
Dataset for fine-tune: IMDB sentiment classification

## Installation

```bash
git clone <repository-url>
cd lora-finetune-pytorch

# Create virtual environment and install dependencies  
uv sync

# Activate the environment
source .venv/bin/activate
```

## LoRA Implementation Details

LoRA decomposes weight updates into low-rank matrices:

- **Original**: `W_new = W_original + ΔW`
- **LoRA**: `W_new = W_original + B·A`, where B·A` is low rank matrix
- **Parameters reduced**: From `d×d` to `d×r + r×d` where `r << d`

```python
# Target Q+V projections in all BERT layers
def apply_lora_to_bert(model, rank=8, alpha=16):
    for layer in model.bert.encoder.layer:
        layer.attention.self.query = LoRALinear(layer.attention.self.query)
        layer.attention.self.value = LoRALinear(layer.attention.self.value)
```

Query and Value projections is usually used in LoRA although the Key and Fully Connected layer can be transformed as well.

# Baseline vs LoRA Performance** (IMDB sentiment classification):
| Metric | Entire Model | LoRA Fine-tuned | Improvement |
| :-- | :-- | :-- | :-- |
| Accuracy | TBD | TBD | TBD% |
| Parameters | TBD | TBD trainable | TBD% reduction |


Lora (best epoch)
[Epoch 3] Validation [Loss Acc]=[0.69 0.68] Training [Loss Acc]=[0.71 0.74] lr=1.00e-0