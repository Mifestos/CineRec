"""
Нагрузочное тестирование API через Locust (для v2.0).

Запуск:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    Открыть http://localhost:8089
"""
import random

from locust import HttpUser, between, task


class RecommenderUser(HttpUser):
    """Имитация поведения реального пользователя в UGC-системе."""
    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Регистрируем нового пользователя и получаем JWT-токен."""
        self.username = f"loaduser_{random.randint(1, 10_000_000)}"
        self.password = "loadpassword123"
        self.email = f"{self.username}@loadtest.com"

        # 1. Регистрация
        self.client.post(
            "/auth/register",
            json={
                "username": self.username,
                "email": self.email,
                "password": self.password
            }
        )

        # 2. Логин для получения токена
        response = self.client.post(
            "/auth/login",
            data={"username": self.username, "password": self.password}
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}

    @task(7)
    def get_recommendations(self):
        """70% трафика: запрос рекомендаций."""
        if not self.token:
            return
        
        top_n = random.choice([5, 10, 20])
        with self.client.get(
            f"/api/v1/recommend?top_n={top_n}",
            headers=self.headers,
            catch_response=True,
            name="/api/v1/recommend"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def rate_movie(self):
        """20% трафика: выставление оценок."""
        if not self.token:
            return
            
        movie_id = random.randint(1, 1000)  # Первые 1000 фильмов точно есть
        score = random.choice([3.0, 4.0, 5.0])
        
        with self.client.post(
            "/ratings/",
            headers=self.headers,
            json={"movie_id": movie_id, "score": score},
            catch_response=True,
            name="/ratings/"
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def write_review(self):
        """10% трафика: написание отзывов (тяжелая операция с NLP)."""
        if not self.token:
            return
            
        movie_id = random.randint(1, 1000)
        text = "Этот фильм оставил у меня очень разные впечатления. Сюжет интересный, но актеры играли не убедительно."
        
        with self.client.post(
            "/reviews/",
            headers=self.headers,
            json={"movie_id": movie_id, "text": text},
            catch_response=True,
            name="/reviews/"
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")