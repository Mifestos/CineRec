from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cinerec.core.config import settings

# Создаем движок (echo=False, чтобы не спамить SQL-запросами в консоль)
engine = create_engine(settings.DATABASE_URL, echo=False)

# Фабрика сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency для FastAPI: выдает сессию и гарантированно закрывает её после запроса."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()