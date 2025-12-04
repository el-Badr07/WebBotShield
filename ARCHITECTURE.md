# WebBotShield - Architecture Documentation

## System Overview

WebBotShield is a distributed real-time bot detection system that analyzes web server logs to identify automated traffic patterns. The system uses stream processing, machine learning-ready feature extraction, and elastic storage for scalability.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion Layer                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
┌───▼────┐                     ┌────▼────┐                   ┌──────▼──────┐
│ Nginx  │                     │ Apache  │                   │   Custom    │
│  Logs  │                     │  Logs   │                   │  App Logs   │
└───┬────┘                     └────┬────┘                   └──────┬──────┘
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
                            ┌───────▼────────┐
                            │   Fluentd      │
                            │  (Log Parser)  │
                            └───────┬────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                        Message Queue Layer                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                            ┌───────▼────────┐
                            │     Kafka      │
                            │ Topic: nginx-  │
                            │      logs      │
                            └───────┬────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    Stream Processing Layer                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   Spark Structured Streaming   │
                    │                                │
                    │  ┌──────────────────────────┐  │
                    │  │  Feature Extraction      │  │
                    │  │  - Burstiness            │  │
                    │  │  - Error Ratios          │  │
                    │  │  - User-Agent Analysis   │  │
                    │  └────────┬─────────────────┘  │
                    │           │                    │
                    │  ┌────────▼─────────────────┐  │
                    │  │  Windowed Aggregations   │  │
                    │  │  - 1-min sliding windows │  │
                    │  │  - Per-IP statistics     │  │
                    │  └────────┬─────────────────┘  │
                    │           │                    │
                    │  ┌────────▼─────────────────┐  │
                    │  │    Bot Scoring Engine    │  │
                    │  │  - Weighted scoring      │  │
                    │  │  - Threshold-based clf   │  │
                    │  └────────┬─────────────────┘  │
                    └────────────────────────────────┘

```
 
## Component Details

### 1. Data Ingestion Layer

#### Fluentd
- **Role**: Log collector and parser
- **Input**: Nginx access logs (tail mode)
- **Processing**:
  - Parses Nginx combined log format
  - Extracts structured fields (IP, timestamp, status, etc.)
  - Adds metadata and enrichment
- **Output**: JSON messages to Kafka

**Key Features**:
- Real-time log tailing
- Buffer management for reliability
- Automatic retry on failures
- Gzip compression for Kafka messages

### 2. Message Queue Layer

#### Apache Kafka
- **Role**: Distributed message broker
- **Topic**: `nginx-logs`
- **Purpose**: Decouples ingestion from processing
- **Benefits**:
  - Buffer for traffic spikes
  - Multiple consumer support
  - Replay capability
  - Fault tolerance

**Configuration**:
- Replication factor: 1 (dev), 3 (prod)
- Retention: 7 days
- Compression: gzip

### 3. Stream Processing Layer

#### Spark Structured Streaming

**Core Applications**: 
- [spark/app/bot_detector.py](spark/app/bot_detector.py) - Rule-based detection
- [spark/app/ml_bot_detector.py](spark/app/ml_bot_detector.py) - ML-based detection

##### 3.1 Data Ingestion
```python
read_kafka_stream()
```
- Consumes from Kafka topic
- Parses JSON messages
- Creates streaming DataFrame
- Handles schema evolution

##### 3.2 Feature Extraction

**Original Features (from logs)**:
- `Port` - HTTP port (80/443)
- `Request_Type` - HTTP/HTTPS (encoded)
- `Protocol` - TCP (encoded)
- `Payload_Size` - Response bytes

**IP Aggregation Features**:
- `requests_per_ip` - Request count per IP in window
- `failure_rate_per_ip` - Error rate per IP
- `avg_payload_per_ip` - Average response size per IP
- `payload_std_per_ip` - Payload size variance

**Temporal Features**:
- `time_since_last_request` - Inter-arrival time
- `inter_arrival_std` - Request timing regularity

**User-Agent Features**:
- `ua_is_empty` - Missing/empty UA detection
- `ua_is_tool` - Bot/tool keyword detection
- `ua_is_browser` - Browser pattern detection

**Request Pattern Features**:
- `is_encrypted` - HTTPS indicator

**Error Features**:
- `is_failure` - HTTP 4xx/5xx indicator
- `protocol_error_rate` - Error rate per protocol

##### 3.3 Windowed Aggregations

**Window Configuration**:
- Window duration: 60 seconds
- Slide duration: 30 seconds
- Watermark: 10 minutes (for late data)

**Aggregations per IP**:
```python
groupBy(window, client_ip).agg(
    count(*).alias("requests_per_ip"),
    avg("is_failure").alias("failure_rate_per_ip"),
    avg("Payload_Size").alias("avg_payload_per_ip"),
    stddev("Payload_Size").alias("payload_std_per_ip"),
    ...
)
```

##### 3.4 Bot Detection Methods

**A. Rule-Based Scoring** (`bot_detector.py`):

Multi-factor scoring with weighted components:

1. **Burst Score** (40% weight):
   ```
   if requests >= threshold:
       score = 1.0
   elif requests >= threshold/2:
       score = (requests - threshold/2) / (threshold/2)
   else:
       score = 0.0
   ```

2. **Error Score** (30% weight):
   ```
   score = min(error_ratio / error_threshold, 1.0)
   ```

3. **User-Agent Score** (30% weight):
   ```
   score = suspicious_ua_count / total_requests
   ```

4. **Final Score**:
   ```
   bot_score = (burst_score × 0.4) + (error_score × 0.3) + (ua_score × 0.3)
   is_bot = bot_score >= 0.7
   ```

**B. ML-Based Detection** (`ml_bot_detector.py`):

Uses a pre-trained Random Forest classifier:

1. **Model**: RandomForestClassifier trained on labeled network traffic
2. **Features**: 16 engineered features (see 3.2)
3. **Training Data**: Time-Series_Network_logs.csv with SMOTE augmentation
4. **Prediction**: Probability threshold ≥ 0.5 → BotAttack

**Model Files** (mounted from `notebook/models/`):
- `webbotshield_bot_detector.joblib` - Trained classifier
- `scaler.joblib` - Feature scaler
- `feature_names.txt` - Feature order reference

**Running ML Detector**:
```bash
# Start with ML profile
docker-compose --profile ml up
```


## Data Flow

### 1. Log Generation
```
Nginx → access.log → Fluentd (tail)
```

### 2. Parsing & Forwarding
```
Fluentd:
  Parse → Enrich → Buffer → Kafka Producer
```

### 3. Stream Processing
```
Spark Streaming:
  Kafka Consumer →
  DataFrame →
  Feature Extraction →
  Windowed Aggregation →
  Scoring 
```

