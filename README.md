# WebBotShield

**Real-time Bot Detection System for Web Server Logs**

WebBotShield is a distributed streaming system that monitors web server logs in real-time to detect bot traffic using Apache Spark Structured Streaming, Kafka, and Elasticsearch. The system analyzes temporal patterns (burstiness), HTTP error ratios, and User-Agent characteristics to distinguish between human and automated traffic.

## Architecture

```
Nginx Logs → Fluentd → Kafka → Spark Streaming → Elasticsearch → Kibana
                                      ↓
                              Bot Detection Engine
                         (Temporal + Error + UA Analysis)
```

### Components

- **Fluentd**: Collects and parses Nginx access logs, forwards to Kafka
- **Apache Kafka**: Message broker for streaming log data
- **Spark Structured Streaming**: Real-time processing and bot detection engine
- **Elasticsearch**: Storage and indexing for logs and detection results
- **Kibana**: Visualization and dashboards for monitoring

## Features

### Detection Algorithms

1. **Burstiness Detection**
   - Monitors request rate per IP in sliding time windows (1 minute)
   - Flags IPs exceeding configurable threshold (default: 100 req/min)

2. **Error Ratio Analysis**
   - Calculates 4xx/5xx error ratios per IP
   - Identifies suspicious patterns (default threshold: 50% error rate)

3. **User-Agent Analysis**
   - Detects missing/empty User-Agents
   - Pattern matching against known bot signatures (curl, wget, scrapy, etc.)
   - Analyzes User-Agent diversity

4. **Weighted Bot Scoring**
   - Combines multiple factors: burstiness (40%) + errors (30%) + UA (30%)
   - Threshold-based classification (default: 0.7 = likely bot)

## Prerequisites

- Docker & Docker Compose
- Python 3.8+ (for log generator)
- At least 8GB RAM recommended
- 10GB free disk space

## Quick Start

### 1. Clone and Setup

```bash
cd webshield
```

### 2. Start Infrastructure

```bash
# Start all services
cd docker
docker-compose up -d

# Check service health
docker-compose ps
```

Services will be available at:
- **Kafka**: localhost:9092
- **Spark Master UI**: http://localhost:8080
- **Elasticsearch**: http://localhost:9200
- **Kibana**: http://localhost:5601

### 3. Create Elasticsearch Indices

```bash
# Wait for Elasticsearch to be ready (30-60 seconds)
bash elasticsearch/setup_indices.sh
```

Or manually:
```bash
curl -X PUT "http://localhost:9200/webshield-logs" -H 'Content-Type: application/json' -d @elasticsearch/mappings.json
curl -X PUT "http://localhost:9200/webshield-detections" -H 'Content-Type: application/json' -d @elasticsearch/mappings.json
```

### 4. Generate Test Logs

```bash
# Install Python dependencies
pip install -r spark/requirements.txt

# Generate continuous logs (5 min with mixed traffic)
python scripts/log_generator.py --mode continuous --duration 300 --normal-rate 5 --bot-rate 2
```

Options:
- `--duration`: Seconds to generate logs (default: 300)
- `--normal-rate`: Normal user requests/second (default: 5)
- `--bot-rate`: Bot requests/second (default: 2)
- `--burst-probability`: Probability of burst attacks (default: 0.05)

### 5. Run Bot Detection

```bash
# Submit Spark job to cluster
docker exec -it webshield-spark-master bash

# Inside container
cd /opt/spark-apps/app
pip install -r ../requirements.txt

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 \
  --master spark://spark-master:7077 \
  bot_detector.py
```

The application will start processing logs and display bot detections in the console.

### 6. View Results in Kibana

1. Open Kibana: http://localhost:5601
2. Go to **Management** → **Stack Management** → **Index Patterns**
3. Create index patterns:
   - `webshield-logs*`
   - `webshield-detections*`
4. Navigate to **Discover** to explore logs and detections
5. Create visualizations and dashboards

## Configuration

Edit [config/settings.yaml](config/settings.yaml) to customize:

```yaml
detection:
  burst_threshold: 100        # Requests/min for burst detection
  error_ratio_threshold: 0.5  # Error ratio threshold
  bot_score_threshold: 0.7    # Bot classification threshold
  window_duration: "1 minute" # Analysis window size
```

## Project Structure

```
webshield/
├── docker/
│   └── docker-compose.yml       # Service orchestration
├── fluentd/
│   ├── Dockerfile               # Fluentd with Kafka plugin
│   └── fluent.conf              # Log parsing configuration
├── spark/
│   ├── app/
│   │   └── bot_detector.py      # Main Spark streaming application
│   └── requirements.txt         # Python dependencies
├── elasticsearch/
│   ├── mappings.json            # Index templates
│   └── setup_indices.sh         # Index creation script
├── kibana/
│   └── dashboards/              # Pre-built dashboards (optional)
├── logs/
│   └── nginx/                   # Nginx log directory
├── scripts/
│   └── log_generator.py         # Test log generator
├── config/
│   └── settings.yaml            # Centralized configuration
└── README.md
```

## Bot Detection Logic

### Scoring Algorithm

For each IP address in a time window, the system calculates:

1. **Burst Score** (0-1):
   - `1.0` if requests >= burst_threshold
   - Linear scaling between threshold/2 and threshold
   - `0.0` if below threshold/2

2. **Error Score** (0-1):
   - Ratio of 4xx/5xx errors to total requests
   - Normalized against error_ratio_threshold

3. **User-Agent Score** (0-1):
   - Ratio of requests with suspicious/missing User-Agents

4. **Final Bot Score**:
   ```
   bot_score = (burst_score × 0.4) + (error_score × 0.3) + (ua_score × 0.3)
   ```

5. **Classification**:
   - `bot_score >= 0.7` → Classified as bot
   - Detection reasons logged for investigation

## Monitoring

### Spark UI
Monitor streaming job health: http://localhost:8080
- Processing rate and latency
- Input/output records
- Batch processing time

### Elasticsearch Indices

```bash
# Check index health
curl http://localhost:9200/_cat/indices?v

# Query detections
curl http://localhost:9200/webshield-detections/_search?pretty

# Get bot alerts
curl -X GET "http://localhost:9200/webshield-detections/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "term": { "is_bot": true }
  }
}'
```

## Troubleshooting

### Logs not flowing to Kafka

```bash
# Check Fluentd logs
docker logs webshield-fluentd

# Verify Kafka topic
docker exec -it webshield-kafka kafka-topics --bootstrap-server localhost:9092 --list
docker exec -it webshield-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic nginx-logs --from-beginning --max-messages 5
```

### Spark job not consuming

```bash
# Check Spark worker logs
docker logs webshield-spark-worker

# Verify Kafka connectivity from Spark
docker exec -it webshield-spark-master bash
telnet kafka 29092
```

### Elasticsearch connection issues

```bash
# Check ES health
curl http://localhost:9200/_cluster/health?pretty

# Check indices
curl http://localhost:9200/_cat/indices?v
```

## Performance Tuning

### For High-Volume Traffic

Edit [docker/docker-compose.yml](docker/docker-compose.yml):

```yaml
spark-worker:
  environment:
    - SPARK_WORKER_MEMORY=4G  # Increase memory
    - SPARK_WORKER_CORES=4    # Increase cores
```

Edit [config/settings.yaml](config/settings.yaml):

```yaml
spark:
  streaming:
    trigger_interval: "10 seconds"  # Faster processing
    max_offsets_per_trigger: 50000  # Higher throughput
```

## Future Enhancements (Phase 2)

- [ ] Machine Learning model integration for advanced detection
- [ ] Feature engineering for ML (session duration, path sequences, etc.)
- [ ] Alerting system (email, Slack, webhooks)
- [ ] Historical data analysis and model training pipeline
- [ ] IP reputation integration
- [ ] Behavioral profiling and anomaly detection
- [ ] Dashboard templates for Kibana

## Development

### Running Tests

```bash
# Unit tests (to be added)
python -m pytest tests/

# Integration tests
python scripts/log_generator.py --mode batch
```

### Adding New Detection Rules

Edit [spark/app/bot_detector.py](spark/app/bot_detector.py):

```python
def custom_detection(self, df):
    """Add your custom detection logic"""
    # Example: Detect sequential path access patterns
    df = df.withColumn("suspicious_pattern",
                       when(col("path").rlike("/admin|/wp-admin"), 1)
                       .otherwise(0))
    return df
```
