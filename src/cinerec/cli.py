import logging
from pathlib import Path
import pandas as pd
import typer
from rich.console import Console
from sqlalchemy import text

from cinerec.db.session import engine, SessionLocal
from cinerec.db.models import Base, User
from cinerec.core.security import get_password_hash

app = typer.Typer(help="CineRec Management CLI")
console = Console()
DATA_DIR = Path("data/raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.command()
def db_init() -> None:
    """Создает схему базы данных (все таблицы)."""
    console.print("[bold yellow]Initializing database schema...[/bold yellow]")
    Base.metadata.create_all(bind=engine)
    console.print("[bold green]Database schema created successfully.[/bold green]")


@app.command()
def db_seed_movies() -> None:
    """Идемпотентно загружает фильмы из MovieLens в PostgreSQL."""
    console.print("[bold yellow]Loading movies into database...[/bold yellow]")
    
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    movies_df = movies_df.rename(columns={"movieId": "id"})
    movies_df["release_year"] = movies_df["title"].str.extract(r"\((\d{4})\)$").astype(float)
    
    # Используем ON CONFLICT DO NOTHING, чтобы не падать, если фильм уже есть
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    def insert_on_conflict(table, conn, keys, data_iter):
        """Кастомный метод для to_sql, игнорирующий дубликаты."""
        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = pg_insert(table.table).values(data).on_conflict_do_nothing(index_elements=["id"])
        conn.execute(stmt)

    movies_df[["id", "title", "genres", "release_year"]].to_sql(
        "movies", 
        engine, 
        if_exists="append", 
        index=False, 
        chunksize=10000, 
        method=insert_on_conflict
    )
    
    # Сбрасываем sequence, чтобы автоинкремент для новых фильмов начался с правильного числа
    with engine.begin() as conn:
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('movies', 'id'), "
            "COALESCE((SELECT MAX(id) FROM movies), 0) + 1, false);"
        ))

    console.print(f"[bold green]Movies loaded and sequence synchronized.[/bold green]")


@app.command()
def create_user(username: str, email: str, password: str) -> None:
    """Создает нового пользователя."""
    with SessionLocal() as db:
        if db.query(User).filter(User.username == username).first():
            console.print(f"[bold red]User {username} already exists.[/bold red]")
            raise typer.Exit(1)
            
        hashed_pwd = get_password_hash(password)
        new_user = User(username=username, email=email, hashed_password=hashed_pwd)
        db.add(new_user)
        db.commit()
        
    console.print(f"[bold green]User {username} created successfully.[/bold green]")


if __name__ == "__main__":
    app()