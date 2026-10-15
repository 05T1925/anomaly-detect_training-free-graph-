import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class ARCZero(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats=1024,
        num_layers=3,
        dropout_rate=0.0,
        activation="ELU",
        num_hops=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.num_hops = num_hops

        if num_layers <= 0:
            raise ValueError("num_layers must be >= 1")

        self.layers.append(nn.Linear(in_feats, h_feats))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(h_feats, h_feats))

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.cross_attn = CrossAttention(h_feats * num_hops)

    def forward(self, graph):
        x_list = graph.x_list

        for i, layer in enumerate(self.layers):
            if i > 0:
                x_list = [self.dropout(x) for x in x_list]
            x_list = [layer(x) for x in x_list]
            if i != len(self.layers) - 1:
                x_list = [self.act(x) for x in x_list]

        x0 = x_list[0]
        residual_list = [xh - x0 for xh in x_list[1:]]
        return torch.hstack(residual_list)


class CrossAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.wq = nn.Linear(embedding_dim, embedding_dim)
        self.wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(self, query_x, support_x):
        q = self.wq(query_x)
        k = self.wk(support_x)
        scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.embedding_dim)
        weights = F.softmax(scores, dim=1)
        return torch.matmul(weights, support_x)

    def get_train_loss(self, x, y, num_prompt):
        positive_indices = torch.nonzero(y == 1).squeeze(1).tolist()
        all_negative_indices = torch.nonzero(y == 0).squeeze(1).tolist()

        if len(positive_indices) == 0 or len(all_negative_indices) == 0:
            return x.sum() * 0.0

        sampled_neg = random.sample(all_negative_indices, min(len(all_negative_indices), len(positive_indices)))

        query_p_embed = x[positive_indices]
        query_n_embed = x[sampled_neg]
        query_embed = torch.vstack([query_p_embed, query_n_embed])

        remain_neg = list(set(all_negative_indices) - set(sampled_neg))
        if len(remain_neg) >= num_prompt:
            support_indices = random.sample(remain_neg, num_prompt)
        elif len(all_negative_indices) >= num_prompt:
            support_indices = random.sample(all_negative_indices, num_prompt)
        else:
            support_indices = random.choices(all_negative_indices, k=num_prompt)

        support_indices = torch.tensor(support_indices, device=y.device)
        support_embed = x[support_indices]

        query_tilde = self.cross_attention(query_embed, support_embed)
        split = len(positive_indices)
        tilde_p_embed, tilde_n_embed = query_tilde[:split], query_tilde[split:]

        yp = torch.ones([len(sampled_neg)], device=y.device)
        yn = -torch.ones([len(positive_indices)], device=y.device)

        loss_qn = F.cosine_embedding_loss(query_n_embed, tilde_n_embed, yp)
        loss_qp = F.cosine_embedding_loss(query_p_embed, tilde_p_embed, yn)
        return torch.mean(loss_qp + loss_qn)

    def get_test_score(self, x, prompt_mask):
        support_indices = torch.nonzero(prompt_mask == 1).squeeze(1).tolist()
        query_indices = torch.nonzero(prompt_mask == 0).squeeze(1).tolist()

        support_embed = x[support_indices]
        query_embed = x[query_indices]

        query_tilde = self.cross_attention(query_embed, support_embed)
        diff = query_embed - query_tilde
        return torch.sqrt(torch.sum(diff ** 2, dim=1))
