"""
Обучение Two-Tower модели (оптимизировано для GPU).

Изменения v3:
- Правильный optimizer (model + loss_fn parameters).
- init_temperature=0.025 → scale=40.
- Диагностика [COS] с gap для мониторинга обучения.
"""
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cinerec.retrieval.two_tower.dataset import TwoTowerDataset, load_movie_features
from cinerec.retrieval.two_tower.model import TwoTowerModel
from cinerec.retrieval.two_tower.loss import InfoNCELoss

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/two_tower")


def compute_item_embedding_cache(
    model: TwoTowerModel,
    content_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Предвычисляет эмбеддинги всех items (no-grad)."""
    model.eval()
    num_items = content_features.size(0)
    all_embeds = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            idx = torch.arange(start, end, device=device)
            content = content_features[start:end].to(device)
            emb = model.forward_item(idx, content)
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


def precompute_user_histories(
    user_to_positives: dict,
    item_emb_cache: torch.Tensor,
    num_users: int,
    embedding_dim: int,
) -> torch.Tensor:
    """ОДИН РАЗ за эпоху вычисляет mean history embedding для каждого user."""
    user_history_embeds = torch.zeros(num_users, embedding_dim)
    for user_idx, positives in user_to_positives.items():
        if positives:
            user_history_embeds[user_idx] = item_emb_cache[positives].mean(dim=0)
    return user_history_embeds


def train_two_tower(
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-2,
    embedding_dim: int = 64,
    device: str = "cuda",
    max_batches: int = None,
    max_samples: int = None,
) -> TwoTowerModel:
    """Обучает Two-Tower модель и сохраняет артефакты."""
    device = torch.device(device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка данных...")
    ds = TwoTowerDataset()
    features, movie_ids, item_counts = load_movie_features()

    logger.info(
        "Num users: %d, Num movies: %d, Content dim: %d",
        ds.num_users, ds.num_movies, features.shape[1],
    )

    # Маппинг content features в порядок ds.movie_ids
    movie_id_to_feature_row = {mid: i for i, mid in enumerate(movie_ids)}
    content_features = torch.tensor(
        features[[movie_id_to_feature_row[mid] for mid in ds.movie_ids]],
        dtype=torch.float32,
    )
    item_counts_ordered = torch.tensor(
        [item_counts[movie_id_to_feature_row[mid]] for mid in ds.movie_ids],
        dtype=torch.float32,
    )

    model = TwoTowerModel(
        num_users=ds.num_users,
        num_movies=ds.num_movies,
        content_dim=features.shape[1],
        embedding_dim=embedding_dim,
    ).to(device)

    # Loss с learnable temperature (scale=40)
    loss_fn = InfoNCELoss(
        item_counts_ordered,
        alpha=0.0,
        init_temperature=0.025,  # scale = 40
    ).to(device)

    # Optimizer видит и параметры модели, и log_temperature
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=learning_rate,
    )

    logger.info(
        "Initial temperature scale: %.2f",
        torch.exp(loss_fn.log_temperature).item(),
    )

    # Сэмплирование датасета (опционально)
    if max_samples and max_samples < len(ds):
        logger.info("Сэмплирование %d из %d пар...", max_samples, len(ds))
        random.seed(42)
        indices = random.sample(range(len(ds)), max_samples)
        ds_subset = Subset(ds, indices)
    else:
        ds_subset = ds

    dataloader = DataLoader(ds_subset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)

        # 1. Пересчёт item-эмбеддингов (no-grad)
        logger.info("Пересчёт кэша item-эмбеддингов...")
        item_emb_cache = compute_item_embedding_cache(model, content_features, device)

        # 2. Предвычисление истории пользователей
        logger.info("Предвычисление user history embeddings...")
        user_history_embeds = precompute_user_histories(
            ds.user_positives, item_emb_cache, ds.num_users, embedding_dim
        )

        model.train()

        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break

            user_idx = batch["user_idx"].to(device)
            pos_idx = batch["positive_idx"].to(device)

            # Индексируем предвычисленную историю
            history_emb = user_history_embeds[user_idx.cpu()].to(device)

            user_emb = model.forward_user(user_idx, history_emb)
            content_batch = content_features[pos_idx.cpu()].to(device)
            item_emb = model.forward_item(pos_idx, content_batch)

            loss = loss_fn(user_emb, item_emb, pos_idx)

            # === ДИАГНОСТИКА [COS] каждые 100 батчей ===
            if epoch == 0 and batch_idx % 100 == 0:
                with torch.no_grad():
                    cosine_matrix = user_emb @ item_emb.T
                    B = cosine_matrix.size(0)
                    diagonal = torch.diagonal(cosine_matrix)
                    mask = ~torch.eye(B, dtype=torch.bool, device=device)
                    off_diag = cosine_matrix[mask].view(B, B - 1).mean(dim=1)
                    logger.info(
                        "[COS] pos=%.4f | neg=%.4f | gap=%.4f | scale=%.2f | loss=%.4f",
                        diagonal.mean().item(),
                        off_diag.mean().item(),
                        (diagonal.mean() - off_diag.mean()).item(),
                        torch.exp(loss_fn.log_temperature).item(),
                        loss.item(),
                    )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            if batch_idx % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = np.mean(epoch_losses)
        logger.info(
            "Epoch %d — avg loss: %.4f (batches: %d)",
            epoch + 1, avg_loss, len(epoch_losses),
        )

    # Сохранение артефактов
    logger.info("Сохраняем артефакты...")
    final_item_emb = compute_item_embedding_cache(model, content_features, device)

    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    np.save(MODEL_DIR / "item_embeddings.npy", final_item_emb.numpy())

    with open(MODEL_DIR / "mappings.pkl", "wb") as f:
        pickle.dump({
            "user_to_idx": ds.user_to_idx,
            "movie_to_idx": ds.movie_to_idx,
            "idx_to_movie": ds.idx_to_movie,
            "content_features": content_features.numpy(),
            "item_counts": item_counts_ordered.numpy(),
        }, f)

    logger.info("Модель сохранена в %s", MODEL_DIR)
    return model