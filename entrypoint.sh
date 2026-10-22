#!/bin/bash
set -e

echo "Starting CineRec API initialization..."

echo "Creating database schema..."
python -m cinerec.cli db-init

echo "Initialization complete. Starting Uvicorn..."
exec uvicorn cinerec.api.main:app --host 0.0.0.0 --port 8000 --reload