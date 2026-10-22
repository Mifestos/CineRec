FROM python:3.12-slim

# Отключаем создание .pyc-файлов и буферизацию вывода (важно для логов в Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Системные зависимости: компиляторы для сборки пакетов и libgomp1 для LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt (кэшируется отдельно)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем копируем код и устанавливаем проект в editable-режиме
COPY . .
RUN pip install --no-cache-dir -e .

# Копируем entrypoint и делаем его исполняемым
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# ENTRYPOINT вызывает entrypoint.sh, который заканчивается на exec "$@"
# Благодаря этому docker-compose.override.yml может подменить команду запуска
ENTRYPOINT ["/app/entrypoint.sh"]

# CMD — команда по умолчанию (production). В dev-override она переопределяется на uvicorn --reload
CMD ["uvicorn", "cinerec.api.main:app", "--host", "0.0.0.0", "--port", "8000"]