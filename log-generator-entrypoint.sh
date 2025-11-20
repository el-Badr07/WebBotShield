#!/bin/bash
set -e

echo "Installing dependencies..."
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git >/dev/null 2>&1
pip install -q pyyaml

echo "Starting continuous log generation with log rotation..."
python /app/scripts/log_generator.py \
  --mode continuous \
  --duration 0 \
  --normal-rate 5 \
  --bot-rate 3 \
  --burst-probability 0.1 \
  --output /app/logs/nginx/access.log \
  --max-file-size-mb 10 \
  --max-backups 3
