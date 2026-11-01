#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "===> Installing Python Dependencies..."
pip install -r requirements.txt

echo "===> Collecting Static Files for Whitenoise..."
python django_app/manage.py collectstatic --noinput

echo "===> Applying Database Migrations..."
python django_app/manage.py migrate

echo "===> Seeding Database with ML Model Benchmarks..."
python django_app/seed_database.py

echo "===> Build Script Completed Successfully!"
