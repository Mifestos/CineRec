import pytest
from fastapi import status


def test_rate_movie_success(client, auth_headers, test_movie):
    """Тест: успешное выставление оценки."""
    response = client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "score": 4.5}
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["movie_id"] == test_movie.id
    assert data["score"] == 4.5
    assert "user_id" in data


def test_rate_movie_update(client, auth_headers, test_movie):
    """Тест: повторная оценка того же фильма обновляет существующую запись."""
    # 1. Ставим оценку 4.5
    client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "score": 4.5}
    )
    
    # 2. Обновляем на 5.0
    response = client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "score": 5.0}
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["score"] == 5.0


def test_rate_movie_not_found(client, auth_headers):
    """Тест: оценка несуществующего фильма возвращает 404."""
    response = client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": 999_999, "score": 4.5}
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_rate_movie_invalid_score_too_high(client, auth_headers, test_movie):
    """Тест: оценка > 5.0 отклоняется Pydantic (422)."""
    response = client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "score": 10.0}
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_rate_movie_invalid_score_too_low(client, auth_headers, test_movie):
    """Тест: оценка < 1.0 отклоняется Pydantic (422)."""
    response = client.post(
        "/ratings/",
        headers=auth_headers,
        json={"movie_id": test_movie.id, "score": 0.5}
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_rate_movie_without_auth(client, test_movie):
    """Тест: без JWT-токена запрос отклоняется (401)."""
    response = client.post(
        "/ratings/",
        json={"movie_id": test_movie.id, "score": 4.5}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED