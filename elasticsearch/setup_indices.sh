#!/bin/bash
# Script to create Elasticsearch indices with proper mappings

ES_HOST="${ES_HOST:-localhost:9200}"

echo "Waiting for Elasticsearch to be ready..."
until curl -s "http://${ES_HOST}/_cluster/health" > /dev/null; do
    sleep 5
done

echo "Elasticsearch is ready!"

# Create webshield-logs index
echo "Creating webshield-logs index..."
curl -X PUT "http://${ES_HOST}/webshield-logs" \
  -H 'Content-Type: application/json' \
  -d '{
  "mappings": {
    "properties": {
      "timestamp": { "type": "date" },
      "client_ip": { "type": "ip" },
      "method": { "type": "keyword" },
      "path": {
        "type": "text",
        "fields": { "keyword": { "type": "keyword" } }
      },
      "status": { "type": "integer" },
      "bytes_sent": { "type": "long" },
      "user_agent": {
        "type": "text",
        "fields": { "keyword": { "type": "keyword" } }
      },
      "referer": {
        "type": "text",
        "fields": { "keyword": { "type": "keyword" } }
      },
      "log_type": { "type": "keyword" }
    }
  }
}'

echo ""
echo "Creating webshield-detections index..."
curl -X PUT "http://${ES_HOST}/webshield-detections" \
  -H 'Content-Type: application/json' \
  -d '{
  "mappings": {
    "properties": {
      "window_start": { "type": "date" },
      "window_end": { "type": "date" },
      "client_ip": { "type": "ip" },
      "request_count": { "type": "long" },
      "error_4xx_count": { "type": "long" },
      "error_5xx_count": { "type": "long" },
      "total_errors": { "type": "long" },
      "total_success": { "type": "long" },
      "suspicious_ua_count": { "type": "long" },
      "avg_bytes_sent": { "type": "double" },
      "error_ratio": { "type": "double" },
      "ua_suspicious_ratio": { "type": "double" },
      "burst_score": { "type": "double" },
      "error_score": { "type": "double" },
      "ua_score": { "type": "double" },
      "bot_score": { "type": "double" },
      "is_bot": { "type": "boolean" },
      "detection_reasons": { "type": "text" },
      "detection_timestamp": { "type": "date" }
    }
  }
}'

echo ""
echo "Indices created successfully!"
