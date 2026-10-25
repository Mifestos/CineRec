# CineRec

[![Fast Tests](https://github.com/Mifestos/CineRec/actions/workflows/fast-tests.yml/badge.svg?branch=main)](https://github.com/Mifestos/CineRec/actions/workflows/fast-tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Full-stack рекомендательная система с поддержкой пользовательского контента (UGC), двухэтапным пайплайном retrieval/ranking и Aspect-Based Sentiment Analysis (ABSA) для обработки отзывов.

## Архитектура

Система решает проблему задержки при скоринге полного каталога (62 тыс. элементов) на каждый запрос путём разделения пайплайна на два этапа:

1. **Retrieval (этап 1).** Матричная факторизация (Implicit ALS) отбирает 50 кандидатов из полного каталога на основе латентных факторов. Сложность — O(log N) за счёт разреженных матричных операций.
2. **Ranking (этап 2).** LightGBM с objective LambdaRank скорит 50 кандидатов по 17 признакам (популярность, средний рейтинг, взаимодействия по жанрам, ALS-скор) и переупорядочивает их по предсказанной релевантности.

### Дополнительные компоненты

- **NLP-слой.** Русскоязычная модель RuBERT выполняет аспектный анализ тональности отзывов. Пайплайн извлекает тональность по конкретным аспектам (сюжет, актёры, музыка), а не единый скор по документу.
- **Cold start fallback.** Пользователи с менее чем 5 оценками получают ранжирование по популярности (топ-50 по взвешенному рейтингу) вместо 404.
- **Кэширование.** Redis кэширует ответы API с TTL 300 секунд. Ключи кэша привязаны к пользователю и параметру `top_n`.
- **Аутентификация.** JWT-аутентификация с OAuth2 password flow. Пароли хэшируются bcrypt. Токен передаётся через `httpOnly` cookie.
- **Reverse Proxy.** Nginx выступает единой точкой входа. Запросы `/api/*` проксируются на FastAPI, остальные — на Next.js.

## Стек

| Слой | Технологии |
|---|---|
| ML / Data | Python 3.12, Pandas, Implicit (ALS), LightGBM (LTR), Scikit-learn, PyArrow |
| NLP | HuggingFace Transformers, RuBERT |
| Backend | FastAPI, Uvicorn, Pydantic v2, python-jose, bcrypt |
| Frontend | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4, shadcn/ui, TanStack Query |
| Хранилище | PostgreSQL 16, SQLAlchemy 2.0, Alembic, Redis 7 |
| DevOps | Docker, Docker Compose, Nginx, GitHub Actions, GHCR |
| Тестирование | Pytest, pytest-mock, pytest-cov, Locust |
## Метрики

| Метрика | Значение |
|---|---|
| Датасет | MovieLens 25M (25 млн оценок, 162 тыс. пользователей, 62 тыс. элементов) |
| Качество ранжировщика (ROC-AUC) | 0.8038 |
| A/B тест NDCG@10 (baseline, LGBMClassifier) | 0.8245 |
| A/B тест NDCG@10 (challenger, LGBMRanker) | 0.8491 |
| Относительный прирост | +2.98% |
| Покрытие тестами (бизнес-логика) | >85% |
| p99 latency (`/recommend`, cached) | <50 мс |
| RPS при 100 конкурентных пользователях | ~45 (узкое место: синхронный NLP-инференс) |

### Известное узкое место

Эндпоинт `/reviews` синхронно вызывает RuBERT, что блокирует event loop FastAPI во время инференса. Это снижает пропускную способность с ~310 RPS (базовая версия без NLP) до ~45 RPS. Решение в следующей итерации — вынос NLP в воркер Celery с брокером Redis для восстановления неблокирующей обработки запросов.

## Структура проекта

Репозиторий использует `src`-layout с разделением слоёв по ответственности.

```text
├── frontend/               # Next.js приложение (UI)
│   ├── app/                # App Router (страницы и layout)
│   ├── components/         # UI-компоненты (shadcn/ui)
│   └── lib/                # API-клиент, Zustand store, типы
├── nginx/                  # Конфигурация Reverse Proxy
├── src/cinerec/            # Backend (Python)
│   ├── api/                # Роутеры FastAPI, схемы Pydantic
│   ├── core/               # Конфигурация, утилиты JWT/security
│   ├── db/                 # ORM-модели SQLAlchemy, CRUD
│   ├── nlp/                # Инференс RuBERT, логика ABSA
│   ├── ranking/            # Feature engineering, обучение LightGBM
│   ├── retrieval/          # Обучение ALS, онлайн-генерация
│   └── pipeline.py         # Оркестратор TwoStageRecommender
├── tests/                  # Pytest тесты (unit, api, integration, evaluation, load)
├── docker-compose.yml      # Оркестрация контейнеров (Nginx, Frontend, API, DB, Redis)
└── Dockerfile              # Сборка бэкенда
```
## Запуск
Требуется Docker Desktop.

### Клонирование и конфигурация окружения:

```bash
git clone https://github.com/Mifestos/CineRec.git
cd CineRec
cp .env.example .env
# Установите SECRET_KEY (сгенерировать: openssl rand -hex 32)
```
Скачайте MovieLens 25M и распакуйте ratings.csv и movies.csv в data/raw/.

### Запуск сервисов:

```bash
docker compose up -d --build
```
Применение миграций, загрузка данных и создание тестового пользователя:
```bash
ddocker compose exec api python -m cinerec.cli db-init
docker compose exec api python -m cinerec.cli db-seed-movies
docker compose exec api python -m cinerec.cli create-user testuser test@example.com strongpassword123
```
Приложение доступно по адресу http://localhost (через Nginx).
Документация OpenAPI (Swagger) — http://localhost/api/docs.

## Тестирование
### Доступные команды (через make):

```bash
make test-fast    # Unit + API тесты (моки, ~3 сек)
make test-eval    # A/B оценка (требует обученных моделей)
make lint         # Статический анализ Ruff
```
### CI-пайплайн
GitHub Actions запускает три workflow:

Fast Tests (на каждый push и PR). Запускает линтинг ruff, затем 22 unit/API теста с порогом покрытия 60%. Пайплайн падает, если не прошёл линтинг или не достигнут порог покрытия.

Scheduled ML Tests (ночной cron). Скачивает датасет MovieLens, обучает ALS и LightGBM с нуля, затем запускает integration и evaluation тесты. Этот workflow не триггерится пушами, чтобы сохранить быструю обратную связь.

Docker Publish (при push в main). Собирает и публикует образ в GitHub Container Registry с тегами latest, sha-<commit> и semver-тегами при наличии.

### Нагрузочное тестирование
Сценарий Locust имитирует 100 конкурентных пользователей с реалистичным распределением запросов:

70% — запросы /recommend

20% — отправка оценок

10% — отправка отзывов (триггерит NLP-инференс)

### Результаты:

Метрика	Значение
Ошибки	0%
RPS	~45
p50 latency	~2000 мс
p95 latency	~6000 мс
Анализ причин роста latency после интеграции NLP см. в разделе «Известное узкое место».

### Трекинг экспериментов
Запуски обучения логируют гиперпараметры, метрики и артефакты в MLflow. Для просмотра экспериментов:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Интерфейс доступен по адресу http://127.0.0.1:5000.

## Roadmap
```text
☑ Переход с офлайн-батчинга на UGC-модель (PostgreSQL)
☑ Добавление JWT-аутентификации и httpOnly cookies
☑ Интеграция NLP-пайплайна на базе ABSA
☑ Внедрение Reverse Proxy (Nginx) и полноценного Frontend (Next.js)
□ Content-Based Fallback для решения проблемы Item Cold Start (новые фильмы)
□ Замена ALS на Two-Tower нейросеть (PyTorch) для улучшения качества Retrieval
□ Вынос NLP-инференса в воркер Celery для восстановления пропускной способности (RPS > 300)
```
## Автор
Mifestos — Data Scientist / ML Engineer

GitHub: @Mifestos