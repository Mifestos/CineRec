from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Глобальная конфигурация приложения."""
    
    # Секреты БЕЗ дефолтных значений (обязательны к заполнению)
    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    
    # Параметры, которые имеют безопасные дефолты
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# При импорте этого модуля, если переменные не заданы, Pydantic выбросит ValidationError
settings = Settings()