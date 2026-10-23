import logging
import re

from transformers import pipeline

logger = logging.getLogger(__name__)

# Загружаем модель один раз при импорте модуля (при старте API)
_sentiment_pipeline = pipeline(
    "sentiment-analysis",  
    model="blanchefort/rubert-base-cased-sentiment",
    device=-1,
    truncation=True,
    max_length=512,
)

# Метки модели -> наши категории
_LABEL_MAP = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
}

# Ключевые слова для ABSA (Aspect-Based Sentiment Analysis)
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "plot": ["сюжет", "история", "сценарий", "поворот", "финал", "концовка", "развязка"],
    "actors": ["актер", "актриса", "игра", "персонаж", "кастинг", "перевоплощение"],
    "music": ["музыка", "саундтрек", "композитор", "мелодия", "звук"],
    "visuals": ["визуал", "эффект", "графика", "картинка", "съемка", "оператор", "кадр"],
    "atmosphere": ["атмосфера", "настроение", "стиль", "режиссура", "погружение"],
}


def _predict(text: str) -> tuple[str, float]:
    """Внутренний вызов модели. Возвращает (категория, уверенность)."""
    try:
        result = _sentiment_pipeline(text[:512])[0]
        label = _LABEL_MAP.get(result["label"].upper(), "neutral")
        return label, float(result["score"])
    except Exception as e:
        logger.error("Sentiment inference failed: %s", e)
        return "neutral", 0.0


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Анализирует общую тональность текста отзыва."""
    if not text or not text.strip():
        return "neutral", 0.0
    return _predict(text)


def analyze_aspects(text: str) -> dict[str, str]:
    """
    Aspect-Based Sentiment Analysis.
    Разбивает отзыв на предложения, для каждого аспекта находит релевантные
    предложения и определяет их тональность.
    """
    if not text or not text.strip():
        return {}

    sentences = re.split(r"(?<=[.!?])\s+", text)
    aspects: dict[str, str] = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        relevant = [
            s for s in sentences
            if any(kw in s.lower() for kw in keywords)
        ]
        if not relevant:
            continue
        combined = " ".join(relevant)
        label, _ = _predict(combined)
        aspects[aspect] = label

    return aspects