from unittest.mock import patch
import pytest
from fastapi import status


@patch("cinerec.db.crud.analyze_sentiment")
@patch("cinerec.db.crud.analyze_aspects")
def test_create_review_success(
    mock_analyze_aspects,
    mock_analyze_sentiment,
    client,
    auth_headers,
    test_movie
):
    """Тест: успешное создание отзыва с NLP-анализом (замоканным)."""
    # Настраиваем моки NLP
    mock_analyze_sentiment.return_value = ("positive", 0.95)
    mock_analyze_aspects.return_value = {"plot": "positive", "actors": "negative"}

    response = client.post(
        "/reviews/",
        headers=auth_headers,
        json={
            "movie_id": test_movie.id,
            "text": "Отличный сюжет, но актеры слабые."  # Длина >= 10
        }
    )

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["movie_id"] == test_movie.id
    assert data["sentiment"] == "positive"
    assert data["sentiment_score"] == 0.95
    assert data["aspects"]["plot"] == "positive"
    assert data["aspects"]["actors"] == "negative"


@patch("cinerec.db.crud.analyze_sentiment")
@patch("cinerec.db.crud.analyze_aspects")
def test_update_existing_review(
    mock_analyze_aspects,
    mock_analyze_sentiment,
    client,
    auth_headers,
    test_movie
):
    """Тест: повторная отправка отзыва на тот же фильм обновляет его."""
    # Первый отзыв
    mock_analyze_sentiment.return_value = ("positive", 0.9)
    mock_analyze_aspects.return_value = {"plot": "positive"}
    client.post(
        "/reviews/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "text": "Хороший фильм, отличный сюжет."}
    )

    # Обновляем отзыв
    mock_analyze_sentiment.return_value = ("negative", 0.8)
    mock_analyze_aspects.return_value = {"plot": "negative"}
    response = client.post(
        "/reviews/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "text": "На самом деле сюжет ужасен и скучен."}
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["sentiment"] == "negative"


def test_create_review_too_short(client, auth_headers, test_movie):
    """Тест: текст отзыва меньше 10 символов отклоняется (422)."""
    response = client.post(
        "/reviews/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "text": "Ужас"}  # < 10 символов
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_review_movie_not_found(client, auth_headers):
    """Тест: отзыв на несуществующий фильм (404)."""
    response = client.post(
        "/reviews/",
        headers=auth_headers,
        json={"movie_id": 999_999, "text": "Этот фильм я никогда не видел, но напишу отзыв."}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_review_without_auth(client, test_movie):
    """Тест: создание отзыва без токена (401)."""
    response = client.post(
        "/reviews/",
        json={"movie_id": test_movie.id, "text": "Анонимный отзыв на фильм."}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@patch("cinerec.db.crud.analyze_sentiment")
@patch("cinerec.db.crud.analyze_aspects")
def test_get_movie_reviews(
    mock_analyze_aspects,
    mock_analyze_sentiment,
    client,
    auth_headers,
    test_movie
):
    """Тест: получение списка отзывов для фильма (доступно без авторизации)."""
    mock_analyze_sentiment.return_value = ("neutral", 0.5)
    mock_analyze_aspects.return_value = {}
    
    # Создаем отзыв
    client.post(
        "/reviews/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "text": "Нормальный фильм, можно посмотреть."}
    )

    # Получаем отзывы БЕЗ токена (эндпоинт публичный)
    response = client.get(f"/reviews/{test_movie.id}")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["movie_id"] == test_movie.id