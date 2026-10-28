"""Smoke test для TwoTowerRetriever — сравнение с ALS."""
import logging
from cinerec.retrieval.two_tower.retriever import TwoTowerRetriever

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    retriever = TwoTowerRetriever(device="cpu")
    
    # Тест: рекомендации для user_id=1 с историей из MovieLens
    test_user = 1
    user_ratings = {1: 5.0, 2: 4.5, 3: 4.0, 4: 5.0, 5: 4.5}  # Toy Story, Jumanji...
    
    recs = retriever.recommend_for_ratings(user_ratings, top_k=10)
    
    print("\n=== Топ-10 рекомендаций для user_id=1 ===")
    for i, r in enumerate(recs, 1):
        print(f"{i}. movieId={r['movieId']}, score={r['score']:.4f}")