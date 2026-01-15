"""
SASRec: Self-Attentive Sequential Recommendation

Paper: "Self-Attentive Sequential Recommendation" - ICDM 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation Model.
    
    Uses Transformer encoder to model sequential patterns in user behavior.
    """
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, num_items)
        
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # Skip padding
        nn.init.xavier_uniform_(self.position_embedding.weight)
        
    def forward(
        self, 
        item_seq: torch.Tensor,  # (batch, seq_len)
        attention_mask: torch.Tensor = None,  # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Args:
            item_seq: Sequence of item indices (0 = padding)
            attention_mask: Mask for padding positions (1 = valid, 0 = pad)
            
        Returns:
            logits: (batch, seq_len, num_items) predictions for each position
        """
        batch_size, seq_len = item_seq.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        item_emb = self.item_embedding(item_seq)  # (batch, seq, dim)
        pos_emb = self.position_embedding(positions)  # (batch, seq, dim)
        
        seq_emb = item_emb + pos_emb
        seq_emb = self.layer_norm(self.dropout(seq_emb))
        
        # Create causal mask (prevent attending to future items)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device) * float('-inf'),
            diagonal=1
        )
        
        # Key padding mask (True = ignore)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = (item_seq == 0)
        
        # Transformer forward
        hidden = self.transformer(
            seq_emb,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        
        # Output logits
        logits = self.output_proj(hidden)  # (batch, seq, num_items)
        
        return logits
    
    def predict(
        self,
        item_seq: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get predictions for the last position in sequence."""
        logits = self.forward(item_seq, attention_mask)
        # Get last non-padding position's prediction
        if attention_mask is not None:
            # Find last valid position for each sequence
            seq_lens = attention_mask.sum(dim=1).long() - 1  # (batch,)
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_idx, seq_lens]  # (batch, num_items)
        else:
            last_logits = logits[:, -1, :]  # (batch, num_items)
        return last_logits


class SASRecTrainer:
    """Trainer for SASRec model."""
    
    def __init__(self, model, optimizer, device, max_seq_len: int = 50):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_seq_len = max_seq_len
        
    def prepare_sequences(
        self,
        user_sequences: Dict[int, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare training sequences for SASRec.
        
        For each user sequence [a, b, c, d], create:
        - Input: [0, a, b, c] (shifted)
        - Target: [a, b, c, d]
        """
        inputs = []
        targets = []
        masks = []
        
        for seq in user_sequences.values():
            if len(seq) < 2:
                continue
                
            # Truncate to max_seq_len
            seq = seq[-self.max_seq_len:]
            seq_len = len(seq)
            
            # Pad sequence
            padded_input = [0] * (self.max_seq_len - seq_len) + [0] + seq[:-1]
            padded_target = [0] * (self.max_seq_len - seq_len) + seq
            mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len
            
            inputs.append(padded_input[-self.max_seq_len:])
            targets.append(padded_target[-self.max_seq_len:])
            masks.append(mask[-self.max_seq_len:])
        
        return (
            torch.LongTensor(inputs),
            torch.LongTensor(targets),
            torch.FloatTensor(masks),
        )
    
    def train_epoch(self, inputs, targets, masks, batch_size=64):
        self.model.train()
        total_loss = 0
        n_samples = inputs.size(0)
        
        perm = torch.randperm(n_samples)
        inputs = inputs[perm].to(self.device)
        targets = targets[perm].to(self.device)
        masks = masks[perm].to(self.device)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            batch_masks = masks[start:end]
            
            self.optimizer.zero_grad()
            
            logits = self.model(batch_inputs, batch_masks)  # (batch, seq, items)
            
            # Cross-entropy loss on valid positions only
            logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, items)
            targets_flat = batch_targets.view(-1)  # (batch*seq,)
            masks_flat = batch_masks.view(-1)  # (batch*seq,)
            
            # Only compute loss on non-padding positions
            valid_idx = (masks_flat > 0) & (targets_flat > 0)
            if valid_idx.sum() == 0:
                continue
                
            loss = F.cross_entropy(
                logits_flat[valid_idx],
                targets_flat[valid_idx],
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(n_batches, 1)


class SequentialRecommender:
    """High-level interface for sequential recommendation."""
    
    def __init__(self, hashtag_to_idx: Dict[str, int], model: SASRec, device: torch.device):
        self.hashtag_to_idx = hashtag_to_idx
        self.idx_to_hashtag = {v: k for k, v in hashtag_to_idx.items()}
        self.model = model
        self.device = device
        self.max_seq_len = model.max_seq_len
        
    def recommend(
        self,
        user_history: List[str],
        exclude_hashtags: List[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Recommend next hashtags based on sequence history."""
        self.model.eval()
        
        # Convert history to indices (+1 because 0 is padding)
        seq_indices = []
        for h in user_history:
            h_lower = h.lower()
            if h_lower in self.hashtag_to_idx:
                seq_indices.append(self.hashtag_to_idx[h_lower] + 1)
        
        if len(seq_indices) == 0:
            return []
        
        # Truncate and pad
        seq_indices = seq_indices[-self.max_seq_len:]
        seq_len = len(seq_indices)
        padded = [0] * (self.max_seq_len - seq_len) + seq_indices
        mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len
        
        with torch.no_grad():
            input_tensor = torch.LongTensor([padded]).to(self.device)
            mask_tensor = torch.FloatTensor([mask]).to(self.device)
            
            logits = self.model.predict(input_tensor, mask_tensor)  # (1, num_items)
            scores = logits.squeeze(0).cpu().numpy()
        
        # Exclude items
        exclude_set = set()
        if exclude_hashtags:
            for h in exclude_hashtags:
                h_lower = h.lower()
                if h_lower in self.hashtag_to_idx:
                    exclude_set.add(self.hashtag_to_idx[h_lower])
        
        # Get top_k
        results = []
        sorted_indices = np.argsort(scores)[::-1]
        for idx in sorted_indices:
            if idx in exclude_set:
                continue
            if idx in self.idx_to_hashtag:
                results.append((self.idx_to_hashtag[idx], float(scores[idx])))
            if len(results) >= top_k:
                break
        
        return results
