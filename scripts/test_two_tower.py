"""Smoke test для Two-Tower модели."""
import torch
from cinerec.retrieval.experimental.two_tower.dataset import TwoTowerDataset, load_movie_features
from cinerec.retrieval.experimental.two_tower.model import TwoTowerModel

ds = TwoTowerDataset()
features, movie_ids = load_movie_features()

model = TwoTowerModel(
    num_users=ds.num_users,
    num_movies=ds.num_movies,
    content_dim=features.shape[1],
)

# Batch из 4 пар
batch = [ds[i] for i in range(4)]
user_idx = torch.stack([b["user_idx"] for b in batch])
positive_idx = torch.stack([b["positive_idx"] for b in batch])

# Заглушка для истории: zeros (на первом шаге у модели нет истории)
history = torch.zeros(4, model.embedding_dim)
content = torch.tensor(features[:4])

user_emb = model.forward_user(user_idx, history)
item_emb = model.forward_item(positive_idx, content)

print("User shape:", user_emb.shape)
print("Item shape:", item_emb.shape)
print("User norms:", user_emb.norm(dim=-1))  # Все должны быть ~1.0
print("Item norms:", item_emb.norm(dim=-1))