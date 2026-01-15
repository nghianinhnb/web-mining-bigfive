
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import logging
import time

class PersonalityLightGCN(nn.Module):
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 64, 
        n_layers: int = 3,
        dropout: float = 0.1,
        user_personality_features: torch.Tensor = None, # (num_users, 5)
        item_content_features: torch.Tensor = None,     # (num_items, content_dim)
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Base Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initializations
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Personality Integration (Optional)
        self.use_personality = user_personality_features is not None
        if self.use_personality:
            # Linear projection from personality (5 dim) to embedding space
            personality_dim = user_personality_features.shape[1]
            self.personality_proj = nn.Linear(personality_dim, embedding_dim)
            self.register_buffer("user_personality", user_personality_features)
            
        # Content Integration (Optional)
        self.use_content = item_content_features is not None
        if self.use_content:
            # Linear projection from content (e.g. 384 dim) to embedding space
            content_dim = item_content_features.shape[1]
            self.content_proj = nn.Linear(content_dim, embedding_dim)
            self.register_buffer("item_content", item_content_features)

    def forward(self, adj_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get final user and item embeddings.
        adj_matrix: Sparse Normalised Adjacency Matrix (N+M, N+M)
        """
        # 1. Initial Embeddings (Layer 0)
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # Add side information if available
        if self.use_personality:
            # Combine learned ID embedding + projected personality
            # E_u = E_id + W_p * P_u
            p_emb = self.personality_proj(self.user_personality)
            u_emb = u_emb + p_emb
            
        if self.use_content:
            # E_i = E_id + W_c * C_i
            c_emb = self.content_proj(self.item_content)
            i_emb = i_emb + c_emb
            
        ego_embeddings = torch.cat([u_emb, i_emb], dim=0) # (N+M, dim)
        all_embeddings = [ego_embeddings]
        
        # 2. Graph Propagation
        for _ in range(self.n_layers):
            # E^(k+1) = D^-1/2 A D^-1/2 E^k
            # Sparse MM
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # 3. Final Aggregation (Mean or Weighted Sum)
        # LightGCN usually takes mean or sum. We use mean here (1/(K+1))
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        final_user_embeddings, final_item_embeddings = torch.split(
            final_embeddings, [self.num_users, self.num_items]
        )
        
        return final_user_embeddings, final_item_embeddings
        
    def get_rating_scores(self, user_indices, final_user_emb, final_item_emb):
        """Compute dot product scores for specific users against all items."""
        users_emb = final_user_emb[user_indices] # (batch, dim)
        scores = torch.matmul(users_emb, final_item_emb.t()) # (batch, num_items)
        return scores


class GNNTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logging.getLogger("GNNTrainer")
        
    def create_adj_matrix(self, user_ids, item_ids, num_users, num_items):
        """
        Create Normalized Adjacency Matrix for LightGCN.
        A = [0, R; R^T, 0]
        L = D^-1/2 A D^-1/2
        """
        # Interactions
        R = coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=(num_users, num_items))
        
        # Adjacency Matrix (N+M, N+M)
        # Row indices: users 0..N-1, items N..N+M-1
        row_idx = np.concatenate([user_ids, item_ids + num_users])
        col_idx = np.concatenate([item_ids + num_users, user_ids])
        data = np.ones(len(row_idx))
        
        adj = coo_matrix((data, (row_idx, col_idx)), shape=(num_users + num_items, num_users + num_items))
        
        # Normalization
        # D_ii = sum(A_ij)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        # L = D^-1/2 * A * D^-1/2
        # SciPy handles diagonal multiplication efficiently
        from scipy.sparse import diags
        d_mat_inv_sqrt = diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Convert to PyTorch Sparse Tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def train_epoch(self, adj_matrix, train_data, batch_size=1024):
        self.model.train()
        total_loss = 0
        
        user_ids = torch.LongTensor(train_data["user_mapping"].values).to(self.device)
        item_ids = torch.LongTensor(train_data["item_mapping"].values).to(self.device)
        
        # Shuffle
        perm = torch.randperm(len(user_ids))
        user_ids = user_ids[perm]
        item_ids = item_ids[perm]
        
        n_batches = (len(user_ids) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(user_ids))
            
            users = user_ids[start:end]
            pos_items = item_ids[start:end]
            
            # Negative Sampling (Random)
            # Simple assumption: random item is likely clear
            # Ideally verify it's not a positive, but at scale it's fine
            neg_items = torch.randint(0, self.model.num_items, (len(users),)).to(self.device)
            
            self.optimizer.zero_grad()
            
            final_u, final_i = self.model(adj_matrix)
            
            u_emb = final_u[users]
            pos_emb = final_i[pos_items]
            neg_emb = final_i[neg_items]
            
            # BPR Loss
            # maximize log(sigmoid(pos_score - neg_score))
            # minimize -log(...)
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            
            # Reg Loss (L2) on initial embeddings
            # reg_loss = (1/2)*(u_emb_0.norm(2).pow(2) + ...)
            # For simplicity, omit or rely on weight_decay in optimizer
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / n_batches


class PersonalitySimGCL(nn.Module):
    """SimGCL: Simple Graph Contrastive Learning for Recommendation.
    
    Key insight: Uses noise perturbation for contrastive views instead of 
    expensive graph augmentation. This is more efficient and robust.
    
    Paper: "Are Graph Augmentations Necessary? Simple Graph Contrastive 
    Learning for Recommendation" - SIGIR 2022
    """
    
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        embedding_dim: int = 64, 
        n_layers: int = 3,
        eps: float = 0.1,  # Noise strength for contrastive views
        user_personality_features: torch.Tensor = None,
        item_content_features: torch.Tensor = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.eps = eps
        
        # Base Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Personality Integration
        self.use_personality = user_personality_features is not None
        if self.use_personality:
            personality_dim = user_personality_features.shape[1]
            self.personality_proj = nn.Linear(personality_dim, embedding_dim)
            self.register_buffer("user_personality", user_personality_features)
            
        # Content Integration
        self.use_content = item_content_features is not None
        if self.use_content:
            content_dim = item_content_features.shape[1]
            self.content_proj = nn.Linear(content_dim, embedding_dim)
            self.register_buffer("item_content", item_content_features)

    def _get_ego_embeddings(self):
        """Get initial embeddings with side information."""
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        if self.use_personality:
            p_emb = self.personality_proj(self.user_personality)
            u_emb = u_emb + p_emb
            
        if self.use_content:
            c_emb = self.content_proj(self.item_content)
            i_emb = i_emb + c_emb
            
        return torch.cat([u_emb, i_emb], dim=0)
    
    def _perturb(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add uniform noise perturbation for contrastive view.
        
        Following SimGCL: noise ~ Uniform(-eps, eps), then L2-normalize
        """
        noise = torch.rand_like(embeddings) * 2 * self.eps - self.eps
        perturbed = embeddings + noise
        return F.normalize(perturbed, p=2, dim=1)

    def forward(
        self, 
        adj_matrix: torch.Tensor,
        perturb: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional perturbation for contrastive views."""
        ego_embeddings = self._get_ego_embeddings()
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            if perturb:
                ego_embeddings = self._perturb(ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        final_user_embeddings, final_item_embeddings = torch.split(
            final_embeddings, [self.num_users, self.num_items]
        )
        
        return final_user_embeddings, final_item_embeddings


class SimGCLTrainer(GNNTrainer):
    """Trainer for SimGCL with InfoNCE contrastive loss."""
    
    def __init__(self, model, optimizer, device, tau: float = 0.2, cl_weight: float = 0.1):
        super().__init__(model, optimizer, device)
        self.tau = tau  # Temperature for InfoNCE
        self.cl_weight = cl_weight  # Weight for contrastive loss
        
    def info_nce_loss(self, view1: torch.Tensor, view2: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss between two views."""
        # Get embeddings for batch
        z1 = view1[indices]  # (batch, dim)
        z2 = view2[indices]  # (batch, dim)
        
        # Normalize
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # Positive pairs: same index across views
        pos_score = torch.sum(z1 * z2, dim=1) / self.tau  # (batch,)
        
        # Negative pairs: all other samples in batch
        # Similarity matrix between z1 and all z2
        neg_score = torch.mm(z1, z2.T) / self.tau  # (batch, batch)
        
        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        # log-sum-exp trick for numerical stability
        loss = -pos_score + torch.logsumexp(neg_score, dim=1)
        
        return loss.mean()
    
    def train_epoch(self, adj_matrix, train_data, batch_size=1024):
        self.model.train()
        total_loss = 0
        total_bpr = 0
        total_cl = 0
        
        user_ids = torch.LongTensor(train_data["user_mapping"].values).to(self.device)
        item_ids = torch.LongTensor(train_data["item_mapping"].values).to(self.device)
        
        perm = torch.randperm(len(user_ids))
        user_ids = user_ids[perm]
        item_ids = item_ids[perm]
        
        n_batches = (len(user_ids) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(user_ids))
            
            users = user_ids[start:end]
            pos_items = item_ids[start:end]
            neg_items = torch.randint(0, self.model.num_items, (len(users),)).to(self.device)
            
            self.optimizer.zero_grad()
            
            # Main view (no perturbation)
            final_u, final_i = self.model(adj_matrix, perturb=False)
            
            # Two perturbed views for contrastive learning
            final_u_p1, final_i_p1 = self.model(adj_matrix, perturb=True)
            final_u_p2, final_i_p2 = self.model(adj_matrix, perturb=True)
            
            # BPR Loss
            u_emb = final_u[users]
            pos_emb = final_i[pos_items]
            neg_emb = final_i[neg_items]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            
            # Contrastive Loss (user-side + item-side)
            unique_users = torch.unique(users)
            unique_items = torch.unique(torch.cat([pos_items, neg_items]))
            
            cl_user_loss = self.info_nce_loss(final_u_p1, final_u_p2, unique_users)
            cl_item_loss = self.info_nce_loss(final_i_p1, final_i_p2, unique_items)
            cl_loss = cl_user_loss + cl_item_loss
            
            # Total loss
            loss = bpr_loss + self.cl_weight * cl_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bpr += bpr_loss.item()
            total_cl += cl_loss.item()
            
        return total_loss / n_batches, total_bpr / n_batches, total_cl / n_batches

