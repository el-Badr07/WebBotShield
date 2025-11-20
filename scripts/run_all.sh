#!/bin/bash
# WebBotShield - Complete Setup and Run Script

set -e

echo "=========================================="
echo "WebBotShield - Automated Setup & Launch"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Start Docker services
echo -e "\n${YELLOW}[1/6] Starting Docker services...${NC}"
cd docker
docker-compose up -d

# Step 2: Wait for services to be ready
echo -e "\n${YELLOW}[2/6] Waiting for services to be ready...${NC}"
echo "Waiting for Kafka..."
sleep 20

echo "Waiting for Elasticsearch..."
until curl -s "http://localhost:9200/_cluster/health" > /dev/null; do
    echo "  Elasticsearch not ready yet, waiting..."
    sleep 5
done
echo -e "${GREEN}Elasticsearch is ready!${NC}"

# Step 3: Create Elasticsearch indices
echo -e "\n${YELLOW}[3/6] Creating Elasticsearch indices...${NC}"
cd ..
bash elasticsearch/setup_indices.sh

# Step 4: Install Python dependencies
echo -e "\n${YELLOW}[4/6] Installing Python dependencies...${NC}"
pip install -r spark/requirements.txt

# Step 5: Start log generator in background
echo -e "\n${YELLOW}[5/6] Starting log generator...${NC}"
python scripts/log_generator.py --mode continuous --duration 600 --normal-rate 5 --bot-rate 2 &
LOG_GEN_PID=$!
echo "Log generator started (PID: $LOG_GEN_PID)"

# Wait a few seconds for logs to start
sleep 5

# Step 6: Submit Spark job
echo -e "\n${YELLOW}[6/6] Submitting Spark bot detection job...${NC}"
echo "Note: You need to run this manually inside the Spark container:"
echo ""
echo "docker exec -it webshield-spark-master bash"
echo "cd /opt/spark-apps/app"
echo "pip install -r ../requirements.txt"
echo "spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 bot_detector.py"
echo ""

echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Services running:"
echo "  - Kafka: localhost:9092"
echo "  - Spark UI: http://localhost:8080"
echo "  - Elasticsearch: http://localhost:9200"
echo "  - Kibana: http://localhost:5601"
echo ""
echo "Next steps:"
echo "  1. Access Kibana at http://localhost:5601"
echo "  2. Create index patterns for 'webshield-*'"
echo "  3. Monitor bot detections in real-time"
echo ""
echo "To stop all services:"
echo "  cd docker && docker-compose down"
echo ""
echo "To stop log generator:"
echo "  kill $LOG_GEN_PID"
