#!/bin/bash
set -e

echo "Starting CineRec API initialization..."

echo "Waiting for database to accept connections..."
until python -c "import psycopg2, os; psycopg2.connect(os.environ['DATABASE_URL'])" 2>/dev/null; do
    echo "Database is unavailable - sleeping 2s"
    sleep 2
done
echo "Database is ready."

echo "Creating database schema..."
python -m cinerec.cli db-init

echo "Initialization complete. Starting Uvicorn..."
exec "$@"