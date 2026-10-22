import random
from locust import HttpUser, task, between

VALID_USER_IDS = list(range(1, 10_001))
INVALID_USER_ID = 999_999_999
TOP_N_OPTIONS = (5, 10, 20)

class RecommenderUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(weight=8)
    def get_recommendations_random_user(self) -> None:
        """Тест на Cache Miss: случайные пользователи."""
        user_id = random.choice(VALID_USER_IDS)
        top_n = random.choice(TOP_N_OPTIONS)
        
        with self.client.get(
            f"/api/v1/recommend/{user_id}?top_n={top_n}", 
            name="/api/v1/recommend/[user_id]?top_n=[top_n]", 
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")

    @task(weight=2)
    def get_recommendations_repeated_user(self) -> None:
        """Тест на Cache Hit: один и тот же пользователь (Redis)."""
        # Здесь URL статичный, поэтому name можно не указывать
        with self.client.get("/api/v1/recommend/1?top_n=10", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")