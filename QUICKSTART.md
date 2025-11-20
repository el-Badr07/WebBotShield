# WebBotShield - Quick Start Guide

This guide will help you get WebBotShield up and running in under 10 minutes.

## Prerequisites Check

```bash
# Verify Docker is installed
docker --version
docker-compose --version

# Verify Python is installed
python --version  # Should be 3.8+
```

## 5-Minute Setup

### Step 1: Start All Services (2 min)

```bash
cd docker
docker-compose up -d
```

Wait for all services to start. You can check status with:
```bash
docker-compose ps
```

All services should show "Up" status.

### Step 2: Wait for Elasticsearch (1 min)

```bash
# Check if Elasticsearch is ready
curl http://localhost:9200/_cluster/health
```

Wait until you see `"status":"yellow"` or `"status":"green"`.

### Step 3: Create Indices (30 sec)

```bash
cd ..
bash elasticsearch/setup_indices.sh
```

Or manually:
```bash
curl -X PUT "http://localhost:9200/webshield-logs"
curl -X PUT "http://localhost:9200/webshield-detections"
```

### Step 4: Generate Test Logs (ongoing)

Open a new terminal and run:

```bash
# Install dependencies
pip install python-dateutil

# Generate logs for 10 minutes
python scripts/log_generator.py --mode continuous --duration 600 --normal-rate 5 --bot-rate 2
```

This will generate:
- 5 normal user requests per second
- 2 bot requests per second
- Random burst attacks every ~20 seconds

### Step 5: Run Spark Bot Detection (1 min)

```bash
# Enter Spark container
docker exec -it webshield-spark-master bash

# Inside the container
cd /opt/spark-apps/app
pip install pyspark kafka-python elasticsearch pyyaml pandas numpy python-dateutil

# Run the detector
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 \
  bot_detector.py
```

You should start seeing bot detections in the console!

## View Results

### Console Output

The Spark job will print detected bots like this:

```
+-------------------+-------------------+-------------+-------------+
|window_start       |window_end         |client_ip    |bot_score    |
+-------------------+-------------------+-------------+-------------+
|2025-01-13 10:00:00|2025-01-13 10:01:00|192.168.1.5  |0.85         |
|2025-01-13 10:00:00|2025-01-13 10:01:00|10.0.0.25    |0.92         |
+-------------------+-------------------+-------------+-------------+
```

### Kibana Dashboard

1. Open Kibana: http://localhost:5601
2. Go to **Management** → **Stack Management** → **Index Patterns**
3. Create pattern: `webshield-detections*`
4. Go to **Discover** tab
5. Filter by `is_bot: true` to see only bot detections

### Direct Elasticsearch Query

```bash
# Get all bot detections
curl -X GET "http://localhost:9200/webshield-detections/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{"query": {"term": {"is_bot": true}}}'

# Count total detections
curl "http://localhost:9200/webshield-detections/_count?pretty"
```

## Understanding the Output

Each detection includes:

- **window_start/end**: Time window analyzed
- **client_ip**: IP address of the detected bot
- **request_count**: Total requests in window
- **error_ratio**: Percentage of 4xx/5xx errors
- **bot_score**: 0-1 score (>0.7 = bot)
- **detection_reasons**: Why it was flagged (e.g., "high_request_rate, suspicious_user_agent")

## Common Issues

### Logs not appearing

```bash
# Check Fluentd
docker logs webshield-fluentd

# Check if logs exist
cat logs/nginx/access.log | head -n 5

# Check Kafka
docker exec -it webshield-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic nginx-logs \
  --from-beginning \
  --max-messages 5
```

### Spark job not starting

```bash
# Check Spark logs
docker logs webshield-spark-master

# Verify Kafka is accessible
docker exec -it webshield-spark-master bash
nc -zv kafka 29092
```

### No bot detections

This is normal if:
- Not enough traffic volume yet (wait 2-3 minutes)
- No burst attacks triggered (they're random)
- Thresholds too high (edit config/settings.yaml)

Lower thresholds for testing:
```yaml
detection:
  burst_threshold: 20  # Lower from 100
  bot_score_threshold: 0.5  # Lower from 0.7
```

## Next Steps

1. **Customize Detection**: Edit `config/settings.yaml` to adjust thresholds
2. **Add Real Logs**: Point Fluentd to your real Nginx logs
3. **Create Dashboards**: Build visualizations in Kibana
4. **Integrate Alerting**: Add webhook/email alerts (Phase 2)

## Stop Everything

```bash
# Stop Spark job: Ctrl+C in Spark terminal

# Stop log generator: Ctrl+C in generator terminal

# Stop all services
cd docker
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```
