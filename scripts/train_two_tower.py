"""CLI для обучения Two-Tower модели на GPU."""
import logging

import torch

from cinerec.retrieval.experimental.two_tower.train import train_two_tower

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    logger.info("Используем GPU: %s", torch.cuda.get_device_name(0))

    # Полное обучение: 2 эпохи × 12,160 батчей = 24,320 шагов.
    # Ожидаемое время: 12-15 минут на GTX 1650 Ti.
    # Ожидаемые результаты: loss < 5.0, gap > 0.15.
    train_two_tower(
        epochs=2,
        batch_size=1024,
        learning_rate=1e-2,
        device="cuda",
        max_batches=None,  
    )