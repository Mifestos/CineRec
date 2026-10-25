from fastapi import status


def test_register_success(client):
    """Тест: успешная регистрация нового пользователя."""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "strongpassword123"
        }
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "new@example.com"
    assert "id" in data
    assert "hashed_password" not in data


def test_register_duplicate_username(client, test_user):
    """Тест: попытка регистрации с уже существующим username (ошибка 400)."""
    response = client.post(
        "/api/auth/register",
        json={
            "username": test_user.username,
            "email": "another@example.com",
            "password": "strongpassword123"
        }
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "already registered" in response.json()["detail"]


def test_register_invalid_email(client):
    """Тест: валидация email через Pydantic (ошибка 422)."""
    response = client.post(
        "/api/auth/register",
        json={
            "username": "newuser",
            "email": "not-an-email",
            "password": "strongpassword123"
        }
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_login_success(client, test_user):
    """Тест: успешный логин, получение JWT-токена."""
    response = client.post(
        "/api/auth/login",
        data={
            "username": test_user.username,
            "password": "password123"
        }
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client, test_user):
    """Тест: логин с неверным паролем (ошибка 401)."""
    response = client.post(
        "/api/auth/login",
        data={
            "username": test_user.username,
            "password": "wrongpassword"
        }
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Incorrect username or password" in response.json()["detail"]


def test_access_protected_route_with_token(client, auth_headers):
    """Тест: доступ к защищенному эндпоинту с валидным токеном."""
    response = client.get("/api/v1/recommend?top_n=5", headers=auth_headers)
    
    assert response.status_code != status.HTTP_401_UNAUTHORIZED


def test_access_protected_route_without_token(client):
    """Тест: доступ к защищенному эндпоинту без токена (ошибка 401)."""
    response = client.get("/api/v1/recommend?top_n=5")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED