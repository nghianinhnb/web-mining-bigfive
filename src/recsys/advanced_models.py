"""
Advanced Recommendation Techniques:
1. Knowledge Graph Embeddings (TransE-style)
2. Hyperbolic Graph Neural Networks (HGCN)

These techniques leverage the hierarchical and relational structure of hashtags.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import math


# ============================================================================
# KNOWLEDGE GRAPH EMBEDDINGS
# ============================================================================

class HashtagKGE(nn.Module):
    """Knowledge Graph Embedding for Hashtag Recommendations.
    
    Learns embeddings where hashtag relationships are modeled as:
    - Co-occurrence relation: h + r_cooc ≈ t
    - Semantic similarity relation: h + r_sim ≈ t
    - User-hashtag relation: u + r_use ≈ h
    
    Based on TransE/RotatE principles.
    """
    
    def __init__(
        self,
        num_users: int,
        num_hashtags: int,
        embedding_dim: int = 64,
        num_relations: int = 3,  # cooc, similarity, personality
        margin: float = 1.0,
        use_rotation: bool = True,  # RotatE vs TransE
    ):
        super().__init__()
        self.num_users = num_users
        self.num_hashtags = num_hashtags
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.use_rotation = use_rotation
        
        # Entity embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.hashtag_embedding = nn.Embedding(num_hashtags, embedding_dim)
        
        # Relation embeddings
        if use_rotation:
            # RotatE: relations are rotations in complex space
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim // 2)
        else:
            # TransE: relations are translations
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.hashtag_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        # Normalize embeddings
        self.hashtag_embedding.weight.data = F.normalize(self.hashtag_embedding.weight.data, p=2, dim=1)
        
    def _rotate(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        """Apply rotation in complex space (RotatE)."""
        # Split into real and imaginary parts
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        
        # Relation as phase angle
        phase = relation
        re_relation = torch.cos(phase)
        im_relation = torch.sin(phase)
        
        # Complex multiplication
        re_result = re_head * re_relation - im_head * im_relation
        im_result = re_head * im_relation + im_head * re_relation
        
        return torch.cat([re_result, im_result], dim=-1)
    
    def score_triplets(
        self,
        heads: torch.Tensor,  # (batch,)
        relations: torch.Tensor,  # (batch,)
        tails: torch.Tensor,  # (batch,)
        head_type: str = "user",  # "user" or "hashtag"
        tail_type: str = "hashtag",
    ) -> torch.Tensor:
        """Score triplets (h, r, t). Lower is better for positive triplets."""
        # Get embeddings
        if head_type == "user":
            h_emb = self.user_embedding(heads)
        else:
            h_emb = self.hashtag_embedding(heads)
            
        if tail_type == "hashtag":
            t_emb = self.hashtag_embedding(tails)
        else:
            t_emb = self.user_embedding(tails)
            
        r_emb = self.relation_embedding(relations)
        
        if self.use_rotation:
            # RotatE: score = ||h ∘ r - t||
            transformed = self._rotate(h_emb, r_emb)
            score = torch.norm(transformed - t_emb, p=2, dim=1)
        else:
            # TransE: score = ||h + r - t||
            score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
            
        return score
    
    def forward(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_tails: torch.Tensor,
        head_type: str = "user",
    ) -> torch.Tensor:
        """Compute margin ranking loss."""
        pos_scores = self.score_triplets(pos_heads, pos_relations, pos_tails, head_type)
        neg_scores = self.score_triplets(pos_heads, pos_relations, neg_tails, head_type)
        
        # Margin ranking loss
        loss = torch.mean(F.relu(self.margin + pos_scores - neg_scores))
        
        return loss
    
    def get_user_hashtag_scores(self, user_idx: int, relation_idx: int = 0) -> np.ndarray:
        """Get scores for all hashtags given a user."""
        with torch.no_grad():
            u_emb = self.user_embedding.weight[user_idx:user_idx+1]  # (1, dim)
            r_emb = self.relation_embedding.weight[relation_idx:relation_idx+1]  # (1, dim or dim/2)
            h_emb = self.hashtag_embedding.weight  # (num_hashtags, dim)
            
            if self.use_rotation:
                u_transformed = self._rotate(u_emb, r_emb)  # (1, dim)
                # Lower distance = better match
                scores = -torch.norm(u_transformed - h_emb, p=2, dim=1)
            else:
                u_translated = u_emb + r_emb  # (1, dim)
                scores = -torch.norm(u_translated - h_emb, p=2, dim=1)
                
            return scores.cpu().numpy()


class KGETrainer:
    """Trainer for Knowledge Graph Embeddings."""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def build_kg_triplets(
        self,
        user_hashtag_interactions: Dict[int, List[int]],
        cooccurrence_matrix: Dict[int, Dict[int, float]],
        threshold: float = 0.1,
    ) -> Tuple[List, List, List]:
        """Build KG triplets from interactions and co-occurrence.
        
        Relations:
        - 0: user_uses_hashtag (u, 0, h)
        - 1: hashtag_cooccurs_with (h1, 1, h2)
        """
        triplets = []
        
        # User-Hashtag triples
        for u_idx, hashtags in user_hashtag_interactions.items():
            for h_idx in hashtags:
                triplets.append(("user", u_idx, 0, h_idx))
        
        # Hashtag-Hashtag co-occurrence triples
        for h1, targets in cooccurrence_matrix.items():
            for h2, prob in targets.items():
                if prob >= threshold and h1 != h2:
                    triplets.append(("hashtag", h1, 1, h2))
        
        return triplets
    
    def train_epoch(self, triplets: List, batch_size: int = 256):
        self.model.train()
        total_loss = 0
        
        # Shuffle
        np.random.shuffle(triplets)
        
        n_batches = (len(triplets) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(triplets))
            batch = triplets[start:end]
            
            # Separate by head type
            user_batch = [t for t in batch if t[0] == "user"]
            hashtag_batch = [t for t in batch if t[0] == "hashtag"]
            
            self.optimizer.zero_grad()
            loss = torch.tensor(0.0, device=self.device)
            
            # User-hashtag triplets
            if user_batch:
                heads = torch.LongTensor([t[1] for t in user_batch]).to(self.device)
                rels = torch.LongTensor([t[2] for t in user_batch]).to(self.device)
                tails = torch.LongTensor([t[3] for t in user_batch]).to(self.device)
                neg_tails = torch.randint(0, self.model.num_hashtags, (len(user_batch),)).to(self.device)
                
                loss = loss + self.model(heads, rels, tails, neg_tails, head_type="user")
            
            # Hashtag-hashtag triplets
            if hashtag_batch:
                heads = torch.LongTensor([t[1] for t in hashtag_batch]).to(self.device)
                rels = torch.LongTensor([t[2] for t in hashtag_batch]).to(self.device)
                tails = torch.LongTensor([t[3] for t in hashtag_batch]).to(self.device)
                neg_tails = torch.randint(0, self.model.num_hashtags, (len(hashtag_batch),)).to(self.device)
                
                loss = loss + self.model(heads, rels, tails, neg_tails, head_type="hashtag")
            
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(n_batches, 1)


# ============================================================================
# HYPERBOLIC GRAPH NEURAL NETWORKS
# ============================================================================

class HyperbolicMath:
    """Math operations in hyperbolic (Poincaré ball) space."""
    
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Möbius addition in Poincaré ball."""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = 1 + 2*c*xy + c**2 * x2 * y2
        
        return num / (denom + 1e-8)
    
    @staticmethod
    def exp_map(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Exponential map from tangent space to Poincaré ball."""
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-8)
        
        return torch.tanh(torch.sqrt(torch.tensor(c)) * v_norm) * v / (torch.sqrt(torch.tensor(c)) * v_norm)
    
    @staticmethod
    def log_map(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Logarithmic map from Poincaré ball to tangent space."""
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=1e-8, max=1-1e-5)
        
        return torch.arctanh(torch.sqrt(torch.tensor(c)) * y_norm) * y / (torch.sqrt(torch.tensor(c)) * y_norm)
    
    @staticmethod
    def project(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Project onto Poincaré ball."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = (1 - eps) / np.sqrt(c)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)
    
    @staticmethod
    def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Compute hyperbolic distance between points."""
        diff = HyperbolicMath.mobius_add(-x, y, c)
        diff_norm = torch.norm(diff, p=2, dim=-1)
        diff_norm = torch.clamp(diff_norm, max=1-1e-5)
        
        return 2 / np.sqrt(c) * torch.arctanh(np.sqrt(c) * diff_norm)


class HyperbolicGCN(nn.Module):
    """Hyperbolic Graph Convolutional Network for Recommendations.
    
    Uses Poincaré ball model to learn embeddings that capture hierarchical
    structure in user-hashtag interactions.
    
    Key insight: Popular/general hashtags near origin, specific ones near boundary.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        n_layers: int = 2,
        curvature: float = 1.0,
        user_personality_features: torch.Tensor = None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.c = curvature
        
        # Embeddings (initialized in tangent space, will be projected)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=0.1)
        
        # Personality integration
        self.use_personality = user_personality_features is not None
        if self.use_personality:
            personality_dim = user_personality_features.shape[1]
            self.personality_proj = nn.Linear(personality_dim, embedding_dim)
            self.register_buffer("user_personality", user_personality_features)
        
        # Layer weights for aggregation
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))
        
    def _get_initial_embeddings(self) -> torch.Tensor:
        """Get initial embeddings projected to hyperbolic space."""
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        if self.use_personality:
            p_emb = self.personality_proj(self.user_personality)
            u_emb = u_emb + 0.1 * p_emb  # Small personality contribution
        
        # Concatenate and project to Poincaré ball
        all_emb = torch.cat([u_emb, i_emb], dim=0)
        all_emb = HyperbolicMath.exp_map(all_emb, self.c)
        all_emb = HyperbolicMath.project(all_emb, self.c)
        
        return all_emb
    
    def forward(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with hyperbolic message passing."""
        # Initial embeddings in hyperbolic space
        h = self._get_initial_embeddings()
        all_embeddings = [h]
        
        for _ in range(self.n_layers):
            # Map to tangent space
            h_tangent = HyperbolicMath.log_map(h, self.c)
            
            # Euclidean aggregation (simpler than full hyperbolic)
            h_agg = torch.sparse.mm(adj_matrix, h_tangent)
            
            # Map back to hyperbolic
            h = HyperbolicMath.exp_map(h_agg, self.c)
            h = HyperbolicMath.project(h, self.c)
            
            all_embeddings.append(h)
        
        # Weighted combination
        weights = F.softmax(self.layer_weights, dim=0)
        final = sum(w * emb for w, emb in zip(weights, all_embeddings))
        final = HyperbolicMath.project(final, self.c)
        
        user_emb, item_emb = torch.split(final, [self.num_users, self.num_items])
        
        return user_emb, item_emb
    
    def compute_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute recommendation scores using hyperbolic distance."""
        # Negative hyperbolic distance as score (closer = higher score)
        scores = -HyperbolicMath.hyperbolic_distance(
            user_emb.unsqueeze(1),  # (n_users, 1, dim)
            item_emb.unsqueeze(0),  # (1, n_items, dim)
            self.c
        )
        return scores


class HyperbolicGCNTrainer:
    """Trainer for Hyperbolic GCN."""
    
    def __init__(self, model, optimizer, device, margin: float = 0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.margin = margin
        
    def create_adj_matrix(self, user_ids, item_ids, num_users, num_items):
        """Same as GNNTrainer.create_adj_matrix"""
        from scipy.sparse import coo_matrix, diags
        
        row_idx = np.concatenate([user_ids, item_ids + num_users])
        col_idx = np.concatenate([item_ids + num_users, user_ids])
        data = np.ones(len(row_idx))
        
        adj = coo_matrix((data, (row_idx, col_idx)), shape=(num_users + num_items, num_users + num_items))
        
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        norm_adj = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)
    
    def train_epoch(self, adj_matrix, train_data, batch_size=512):
        self.model.train()
        total_loss = 0
        
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
            
            user_emb, item_emb = self.model(adj_matrix)
            
            u_emb = user_emb[users]
            pos_emb = item_emb[pos_items]
            neg_emb = item_emb[neg_items]
            
            # Hyperbolic distances
            pos_dist = HyperbolicMath.hyperbolic_distance(u_emb, pos_emb, self.model.c)
            neg_dist = HyperbolicMath.hyperbolic_distance(u_emb, neg_emb, self.model.c)
            
            # Margin ranking loss
            loss = torch.mean(F.relu(self.margin + pos_dist - neg_dist))
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / n_batches
