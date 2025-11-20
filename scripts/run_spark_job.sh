#!/bin/bash
# Script to run Spark bot detection job inside the Spark container

echo "Installing Python dependencies..."
pip install -q pyspark kafka-python elasticsearch pyyaml pandas numpy python-dateutil

echo "Starting Spark Structured Streaming job..."
spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 \
  --conf spark.executor.memory=2g \
  --conf spark.driver.memory=1g \
  --conf spark.sql.shuffle.partitions=4 \
  /opt/spark-apps/app/bot_detector.py
