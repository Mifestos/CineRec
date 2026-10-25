from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from cinerec.core.config import settings
from cinerec.db import crud
from cinerec.db.models import User
from cinerec.db.session import get_db

# auto_error=False позволяет нам вручную обрабатывать отсутствие заголовка
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(
    request: Request,
    token_header: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Извлекает текущего пользователя из JWT-токена (куки или заголовок)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # 1. Пытаемся получить токен из Cookie (для фронтенда)
    token = request.cookies.get("access_token")
    
    # 2. Если куки нет, берем из заголовка (для Swagger UI)
    if not token:
        token = token_header
        
    if not token:
        raise credentials_exception

    # Очищаем от возможных кавычек (curl сохраняет значение как "Bearer ...")
    token = token.strip('"').strip("'")
    
    try:
        # Отделяем схему 'Bearer' от самого токена
        scheme, _, param = token.partition(" ")
        if scheme.lower() == "bearer":
            token = param
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except (JWTError, ValueError) as e:
        raise credentials_exception from e

    user = crud.get_user_by_id(db, user_id=user_id)
    if user is None:
        raise credentials_exception
    return user