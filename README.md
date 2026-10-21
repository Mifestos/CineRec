# CineRec

![Tests](https://github.com/Mifestos/CineRec/actions/workflows/python-tests.yml/badge.svg?branch=main)

Production-ready рекомендательная система для стримингового сервиса, построенная на базе датасета MovieLens 25M. Реализует классическую двухэтапную архитектуру, используемую в индустрии (Netflix, Spotify, Яндекс): Retrieval (генерация кандидатов) и Ranking (персонализированное ранжирование).

## Бизнес-ценность и архитектура

В высоконагруженных системах невозможно прогнать всю базу контента (62 000 фильмов) через тяжелую ML-модель за 50 миллисекунд. Проект решает эту проблему с помощью двух этапов:

1. **Retrieval (Stage 1):** Модель матричной факторизации (ALS) быстро отбирает 50 наиболее релевантных кандидатов из 62 000 фильмов на основе скрытых факторов.
2. **Ranking (Stage 2):** Градиентный бустинг (LightGBM) применяет десятки признаков (популярность фильма, средний рейтинг, жанры, ALS-скор) к 50 кандидатам, вычисляя вероятность того, что пользователю понравится фильм.

**Дополнительные возможности:**
* **Кэширование:** Redis используется для кэширования ответов API, что снижает нагрузку на ML-модели при повторных запросах.
* **Cold Start Fallback:** Если пользователь отсутствует в обучающей выборке, система возвращает витрину популярных фильмов (Top-50 по среднему рейтингу), предотвращая ошибки 404.

**Стек технологий:**
* **ML:** Python 3.12, Implicit (ALS), LightGBM, Pandas, Scikit-learn
* **Backend:** FastAPI, Uvicorn, Pydantic, Redis
* **DevOps:** Docker, Docker Compose
* **Тестирование:** Pytest, Pytest-mock, Pytest-cov

## Результаты (Metrics)

* **Датасет:** MovieLens 25M (25 млн оценок, 162 000 пользователей, 62 000 фильмов)
* **Качество ранжировщика:** ROC-AUC **0.8038**
* **Время ответа API (Inference):** < 50 мс на запрос (без учета cold start)
* **Покрытие тестами (API):** > 85%
* **Потребление памяти:** ~2.5 GB (загрузка весов моделей и справочников)

## A/B Тестирование (Offline)

Для проверки эффективности Learning-to-Rank алгоритма был проведен офлайн A/B тест на 500 случайных пользователях (метрика NDCG@10).

* **Group A (Baseline):** LGBMClassifier (Бинарная классификация) — NDCG@10: **0.8245**
* **Group B (Challenger):** LGBMRanker (Learning-to-Rank, Lambdarank) — NDCG@10: **0.8491**
* **Прирост качества (Lift):** **+2.98%**

## Быстрый старт (Docker)

Для запуска проекта требуется установленный Docker Desktop.

Клонируйте репозиторий:
```bash
git clone https://github.com/Mifestos/CineRec.git
cd CineRec
```
Подготовьте данные (скачайте MovieLens 25M) и распакуйте ratings.csv, movies.csv в папку data/raw/.

Обучите модели (опционально, если нет готовых весов):

```bash
pip install -r requirements.txt
python src/cinerec/retrieval/train_als.py
python src/cinerec/ranking/features.py
python src/cinerec/ranking/train_ranker.py
```
Запустите сервис:

```bash
docker compose up -d --build
```
```text
API будет доступно по адресу http://localhost:8000. 
Документация Swagger: http://localhost:8000/docs.
```

Пример запроса:

```bash
curl "http://localhost:8000/api/v1/recommend/1?top_n=5"
```
Пример ответа:
```json
[
  {
    "movieId": 3089,
    "title": "Bicycle Thieves (a.k.a. The Bicycle Thief) (a.k.a. The Bicycle Thieves) (Ladri di biciclette) (1948)",
    "genres": "Drama",
    "score": 0.8550311239048133
  },
  {
    "movieId": 1232,
    "title": "Stalker (1979)",
    "genres": "Drama|Mystery|Sci-Fi",
    "score": 0.8548460316116894
  }
]
```
## Тестирование
Проект покрыт тестами с использованием pytest. Для быстрой проверки логики API (без загрузки тяжелых ML-моделей) используются моки (unittest.mock).

Unit-тесты: Проверка успешных ответов, валидации параметров, обработки кэша и cold start fallback.

Integration-тесты: Проверка реального пайплайна ALS + LightGBM на соответствие формату.

Запуск быстрых тестов (без загрузки моделей):

```bash
pytest -m "not integration"
```
## Нагрузочное тестирование

Проект протестирован с помощью Locust (100 одновременных пользователей, Ramp-up 10/сек).
Модели и Redis запущены в Docker.

**Результаты:**
* **Пропускная способность (RPS):** ~310 запросов в секунду (пиковая), ~174 RPS (стабильная).
* **Время ответа (p50):** 8 мс
* **Время ответа (p95):** 26 мс
* **Время ответа (p99):** 45 мс
* **Ошибки:** 0%

## Структура проекта
Проект использует src-layout для изоляции кода от данных и конфигураций.

```text
├── data/                  # Сырые данные и Parquet-файлы с признаками
├── models/                # Веса обученных моделей (ALS, LightGBM)
├── src/cinerec/
│   ├── api/               # FastAPI приложение (main, routes, schemas, cache)
│   ├── ranking/           # Feature Engineering и обучение LightGBM
│   ├── retrieval/         # Обучение и инференс ALS
│   └── pipeline.py        # Класс TwoStageRecommender (объединяет оба этапа)
├── tests/                 # Unit и Integration тесты (Pytest)
├── Dockerfile             # Инструкция сборки образа
├── docker-compose.yml     # Оркестрация контейнеров (API + Redis)
└── requirements.txt       # Зависимости (с фиксированными версиями)
```
## Планы по развитию (Roadmap)
```text
□ Внедрение CI/CD (GitHub Actions) для автоматического запуска тестов при пуше.
□ Проведение A/B-тестирования (эмуляция онлайн-трафика) для оценки бизнес-метрик.
□ Переход от классического ALS к Two-Tower Neural Network (PyTorch) для улучшения качества Retrieval.
```
## Автор
Mifestos

