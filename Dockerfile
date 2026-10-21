FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости (компиляторы для сборки библиотек и libgomp1 для LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements.txt для кэширования слоя (ускоряет повторную сборку)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код проекта
COPY . .
RUN pip install --no-cache-dir -e .

# Открываем порт
EXPOSE 8000

# Запускаем сервер через uvicorn (без флага --reload, так как это продакшн-режим)
CMD ["uvicorn", "cinerec.api.main:app", "--host", "0.0.0.0", "--port", "8000"]