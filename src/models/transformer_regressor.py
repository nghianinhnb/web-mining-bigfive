#!/usr/bin/env python3
"""Transformer-based regressor for Big Five personality prediction with learning curves."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.config import MODELS_DIR, RESULTS_DIR, TRAIT_COLS, ENCODER_MODEL


class PersonalityDataset(Dataset):
    """Dataset for personality prediction from concatenated tweets."""
    
    def __init__(
        self,
        texts: List[str],
        targets: Optional[np.ndarray],
        tokenizer,
        max_length: int = 512,
        max_chunks: int = 8,
    ):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize without truncation first
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        
        # Calculate total chunks needed
        # We reserve space for [CLS] and [SEP] in each chunk (total -2)
        chunk_len = self.max_length - 2
        total_len = len(input_ids)
        num_chunks = min(
            (total_len + chunk_len - 1) // chunk_len,
            self.max_chunks
        )
        if num_chunks == 0:
            num_chunks = 1
            
        final_input_ids = torch.zeros((self.max_chunks, self.max_length), dtype=torch.long)
        final_attention_mask = torch.zeros((self.max_chunks, self.max_length), dtype=torch.long)
        
        # Prepare chunks
        for i in range(num_chunks):
            start = i * chunk_len
            end = min(start + chunk_len, total_len)
            
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            
            # Add [CLS] and [SEP]
            # Assumes tokenizer has cls_token_id and sep_token_id
            cls_id = torch.tensor([self.tokenizer.cls_token_id])
            sep_id = torch.tensor([self.tokenizer.sep_token_id])
            
            chunk_ids = torch.cat([cls_id, chunk_ids, sep_id])
            chunk_mask = torch.cat([torch.ones(1), chunk_mask, torch.ones(1)])
            
            # Pad
            padding_len = self.max_length - len(chunk_ids)
            if padding_len > 0:
                chunk_ids = torch.cat([chunk_ids, torch.tensor([self.tokenizer.pad_token_id] * padding_len)])
                chunk_mask = torch.cat([chunk_mask, torch.zeros(padding_len)])
            
            final_input_ids[i] = chunk_ids
            final_attention_mask[i] = chunk_mask
            
        item = {
            "input_ids": final_input_ids,  # (max_chunks, max_length)
            "attention_mask": final_attention_mask,
        }
        
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return item


class PersonalityRegressor(nn.Module):
    """Transformer encoder with linear head for personality trait prediction."""
    
    def __init__(self, model_name: str, num_traits: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, num_traits)
    
    def forward(self, input_ids, attention_mask):
        # input_ids: (Batch, Chunks, Seq)
        batch_size, num_chunks, seq_len = input_ids.size()
        
        # Flatten chunks into batch dimension
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)
        
        outputs = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        )
        
        # Mean pooling over sequence (excluding padding)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask_flat.unsqueeze(-1).expand(last_hidden.size()).float()
        
        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_chunks = sum_hidden / sum_mask  # (Batch*Chunks, Hidden)
        
        # Reshape back to (Batch, Chunks, Hidden)
        pooled_chunks = pooled_chunks.view(batch_size, num_chunks, -1)
        
        # Mean pooling over chunks
        # Only consider chunks that are not fully padding (though attention mask handles this mostly)
        # We can use the sum of attention masks per chunk to determine valid chunks
        chunk_lens = attention_mask.sum(dim=-1) # (Batch, Chunks)
        chunk_mask = (chunk_lens > 0).float().unsqueeze(-1) # (Batch, Chunks, 1)
        
        sum_pooled = torch.sum(pooled_chunks * chunk_mask, dim=1)
        sum_chunk_mask = chunk_mask.sum(dim=1).clamp(min=1e-9)
        
        final_pooled = sum_pooled / sum_chunk_mask # (Batch, Hidden)
        
        final_pooled = self.dropout(final_pooled)
        return self.regressor(final_pooled)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_accuracy_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    tolerance: float = 0.1,
) -> Dict[str, float]:
    """Compute accuracy metrics for regression.
    
    Args:
        predictions: Predicted values
        targets: True values
        tolerance: Threshold for considering a prediction "correct"
    
    Returns:
        Dict with accuracy metrics:
        - tolerance_acc: Percentage of predictions within tolerance
        - pearson_r: Average Pearson correlation across traits
    """
    # Tolerance-based accuracy (predictions within threshold)
    within_tolerance = np.abs(predictions - targets) <= tolerance
    tolerance_acc = np.mean(within_tolerance)
    
    # Pearson correlation per trait
    correlations = []
    for i in range(predictions.shape[1]):
        r, _ = pearsonr(predictions[:, i], targets[:, i])
        if not np.isnan(r):
            correlations.append(r)
    avg_pearson = np.mean(correlations) if correlations else 0.0
    
    return {
        "tolerance_acc": tolerance_acc,
        "pearson_r": avg_pearson,
    }


class TransformerTrainer:
    """Trainer for transformer-based personality regressor with learning curves."""
    
    def __init__(
        self,
        model_name: str = ENCODER_MODEL,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        max_length: int = 512,
        max_chunks: int = 8,
        device: Optional[str] = None,
        results_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = PersonalityRegressor(model_name).to(self.device)
        
        self.training_history = []
        self.best_model_state = None
        self.best_eval_loss = float("inf")
    
    def fit(
        self,
        train_texts: List[str],
        train_targets: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        use_cosine_schedule: bool = True,
        warmup_epochs: int = 2,
        save_suffix: str = "",
    ) -> "TransformerTrainer":
        """Train the model with learning curve logging, early stopping, and cosine LR decay.
        
        Args:
            train_texts: Training texts
            train_targets: Training labels
            val_texts: Validation texts
            val_targets: Validation labels
            epochs: Maximum training epochs
            early_stopping_patience: Epochs to wait before early stopping
            use_cosine_schedule: Whether to use cosine annealing LR scheduler
            warmup_epochs: Number of warmup epochs before cosine decay
            save_suffix: Suffix for saved learning curve files (e.g., language code)
        """
        
        train_dataset = PersonalityDataset(
            train_texts, train_targets, self.tokenizer, self.max_length, self.max_chunks
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        
        val_loader = None
        if val_texts is not None and val_targets is not None:
            val_dataset = PersonalityDataset(
                val_texts, val_targets, self.tokenizer, self.max_length, self.max_chunks
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
            )
        
        # Layer-wise Learning Rate Decay (LLRD)
        optimizer_grouped_parameters = self._get_optimizer_params(self.learning_rate)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Cosine annealing scheduler
        scheduler = None
        if use_cosine_schedule:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=self.learning_rate * 0.01,  # Min LR = 1% of initial
            )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode="min",
        )
        
        self.training_history = []
        
        for epoch in range(epochs):
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Warmup: linear increase
            if epoch < warmup_epochs:
                warmup_lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
                current_lr = warmup_lr
            
            # Training phase
            self.model.train()
            train_losses = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
            
            avg_train_loss = np.mean(train_losses)
            
            # Evaluation phase
            eval_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "learning_rate": current_lr,
            }
            
            if val_loader is not None:
                eval_loss, eval_rmse, eval_mae, eval_acc, eval_pearson = self._evaluate_loader(
                    val_loader, criterion
                )
                eval_metrics["eval_loss"] = eval_loss
                eval_metrics["eval_rmse"] = eval_rmse
                eval_metrics["eval_mae"] = eval_mae
                eval_metrics["eval_acc"] = eval_acc
                eval_metrics["eval_pearson"] = eval_pearson
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                
                print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                      f"eval_loss={eval_loss:.4f}, eval_rmse={eval_rmse:.4f}, "
                      f"eval_mae={eval_mae:.4f}, eval_acc={eval_acc:.4f}, "
                      f"pearson_r={eval_pearson:.4f}, lr={current_lr:.2e}")
                
                # Check early stopping
                if early_stopping(eval_loss, epoch + 1):
                    print(f"Early stopping triggered at epoch {epoch + 1}. "
                          f"Best epoch: {early_stopping.best_epoch}")
                    break
            else:
                print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, lr={current_lr:.2e}")
            
            self.training_history.append(eval_metrics)
            
            # Update scheduler after warmup
            if scheduler is not None and epoch >= warmup_epochs:
                scheduler.step()
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model from epoch {early_stopping.best_epoch}")
        
        # Save learning curves
        self._save_learning_curves(suffix=save_suffix)
        
        return self
    
    def _evaluate_loader(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, float, float, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        avg_loss = total_loss / len(all_targets)
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        mae = np.mean(np.abs(all_preds - all_targets))
        
        # Compute accuracy metrics
        acc_metrics = compute_accuracy_metrics(all_preds, all_targets, tolerance=0.1)
        
        return avg_loss, rmse, mae, acc_metrics["tolerance_acc"], acc_metrics["pearson_r"]
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict personality traits for input texts."""
        self.model.eval()
        
        dataset = PersonalityDataset(texts, None, self.tokenizer, self.max_length, self.max_chunks)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                all_preds.append(outputs.cpu().numpy())
        
        return np.concatenate(all_preds, axis=0)
    
    def evaluate(
        self,
        texts: List[str],
        targets: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, float]:
        """Evaluate model and return metrics per trait."""
        predictions = self.predict(texts)
        y_true = np.array(targets)
        
        metrics = {}
        
        # Per-trait RMSE and MAE
        for i, trait in enumerate(TRAIT_COLS):
            trait_name = trait.replace("y_", "")
            rmse = np.sqrt(np.mean((y_true[:, i] - predictions[:, i]) ** 2))
            mae = np.mean(np.abs(y_true[:, i] - predictions[:, i]))
            
            # Per-trait accuracy (tolerance-based)
            acc = np.mean(np.abs(y_true[:, i] - predictions[:, i]) <= 0.1)
            
            # Per-trait Pearson correlation
            r, _ = pearsonr(predictions[:, i], y_true[:, i])
            
            metrics[f"rmse_{trait_name}"] = rmse
            metrics[f"mae_{trait_name}"] = mae
            metrics[f"acc_{trait_name}"] = acc
            metrics[f"pearson_{trait_name}"] = r if not np.isnan(r) else 0.0
        
        # Average metrics
        metrics["avg_rmse"] = np.mean([
            metrics[f"rmse_{t.replace('y_', '')}"] for t in TRAIT_COLS
        ])
        metrics["avg_mae"] = np.mean([
            metrics[f"mae_{t.replace('y_', '')}"] for t in TRAIT_COLS
        ])
        metrics["avg_acc"] = np.mean([
            metrics[f"acc_{t.replace('y_', '')}"] for t in TRAIT_COLS
        ])
        metrics["avg_pearson"] = np.mean([
            metrics[f"pearson_{t.replace('y_', '')}"] for t in TRAIT_COLS
        ])
        
        return metrics
    
    def _save_learning_curves(self, suffix: str = "") -> None:
        """Save training history and plot learning curves."""
        if not self.training_history:
            return
            
    def _save_learning_curves(self, suffix: str = "") -> None:
        """Save training history and plot learning curves."""
        if not self.training_history:
            return
            
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df = pd.DataFrame(self.training_history)
        
        # Use suffix (e.g., language code) in filenames
        fname_suffix = f"_{suffix}" if suffix else ""
        csv_path = self.results_dir / f"learning_curve{fname_suffix}.csv"
        df.to_csv(csv_path, index=False)
        
        # Plot learning curves (2 subplots)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = df["epoch"].values
        
        # Left plot: Loss curves
        ax1 = axes[0]
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(epochs, df["train_loss"], "b-", label="Train Loss", marker="o", markersize=3)
        if "eval_loss" in df.columns:
            ax1.plot(epochs, df["eval_loss"], "r-", label="Eval Loss", marker="s", markersize=3)
        ax1.legend()
        ax1.set_title(f"Loss Curves{f' ({suffix})' if suffix else ''}")
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Metrics (RMSE, Accuracy, Pearson)
        ax2 = axes[1]
        ax2.set_xlabel("Epoch")
        if "eval_rmse" in df.columns:
            ax2.plot(epochs, df["eval_rmse"], "g-", label="RMSE", marker="^", markersize=3)
        if "eval_acc" in df.columns:
            ax2.plot(epochs, df["eval_acc"], "m-", label="Accuracy (tol=0.1)", marker="d", markersize=3)
        if "eval_pearson" in df.columns:
            ax2.plot(epochs, df["eval_pearson"], "c-", label="Pearson r", marker="x", markersize=3)
        ax2.legend()
        ax2.set_title(f"Evaluation Metrics{f' ({suffix})' if suffix else ''}")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Training Learning Curves{f' - {suffix}' if suffix else ''}", fontsize=14)
        plt.tight_layout()
        
        png_path = self.results_dir / f"learning_curve{fname_suffix}.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved learning curve to {csv_path} and {png_path}")
    
    def _get_optimizer_params(self, base_lr: float, weight_decay: float = 0.01, decay_factor: float = 0.95):
        """Apply Layer-wise Learning Rate Decay."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = []
        
        # 1. Regressor Head (Top layers) - uses base_lr
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.regressor.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": base_lr
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.regressor.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": base_lr
        })
        
        # 2. Encoder Layers (Decay downwards)
        # We assume standard transformer structure (embeddings + encoder.layer.0...N)
        # Get all encoder named parameters
        encoder_params = list(model.encoder.named_parameters())
        
        # Group by layer index
        layers = {}
        embeddings = []
        others = []
        
        for n, p in encoder_params:
            if "embeddings" in n:
                embeddings.append((n, p))
            elif "layer." in n:
                # Extract layer index
                try:
                    parts = n.split(".")
                    layer_idx = int(parts[parts.index("layer") + 1])
                    if layer_idx not in layers:
                        layers[layer_idx] = []
                    layers[layer_idx].append((n, p))
                except (ValueError, IndexError):
                    others.append((n, p))
            else:
                others.append((n, p))
                
        # Assign LRs
        # Max layer index
        if layers:
            max_layer = max(layers.keys())
        else:
            max_layer = 0
            
        # Layers 0..max_layer. LLRD means layer N gets base_lr, layer N-1 gets base_lr * decay...
        for layer_idx in sorted(layers.keys(), reverse=True):
            lr = base_lr * (decay_factor ** (max_layer - layer_idx + 1))
            
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layers[layer_idx] if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layers[layer_idx] if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr
            })
            
        # Embeddings (Lowest LR)
        lr_embed = base_lr * (decay_factor ** (max_layer + 2))
        optimizer_grouped_parameters.append({
            "params": [p for n, p in embeddings if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr_embed
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in embeddings if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr_embed
        })
        
        # Others
        lr_others = base_lr * decay_factor
        if others:
            optimizer_grouped_parameters.append({
                "params": [p for n, p in others if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr_others
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in others if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr_others
            })
            
        return optimizer_grouped_parameters

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model checkpoint."""
        if path is None:
            path = MODELS_DIR / "transformer_checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "training_history": self.training_history,
        }, path)
        
        print(f"Saved model checkpoint to {path}")
        return path
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TransformerTrainer":
        """Load model from checkpoint."""
        if path is None:
            path = MODELS_DIR / "transformer_checkpoint.pt"
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        trainer = cls(
            model_name=checkpoint["model_name"],
            learning_rate=checkpoint["learning_rate"],
            batch_size=checkpoint["batch_size"],
            max_length=checkpoint["max_length"],
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.to(trainer.device)
        trainer.training_history = checkpoint.get("training_history", [])
        
        return trainer
