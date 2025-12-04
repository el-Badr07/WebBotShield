# Kibana Integration Guide for WebBotShield

## Overview

Kibana is now fully integrated with Elasticsearch and ready to visualize your WebBotShield data. This guide explains how to:
1. Access the data view in Kibana Analytics
2. Set up Filebeat integration for log collection
3. Configure the complete monitoring dashboard

---

## 1. ✅ Data View Setup (COMPLETED)

### What was created:
- **Data View Name**: "Detections"
- **Index Pattern**: `webshield-detections*`
- **Time Field**: `detection_timestamp`
- **Status**: ✅ Active and ready to use

### Access the Data View in Kibana:

#### Option A: Direct URL
```
http://localhost:5601/app/discover
```

#### Option B: Through Kibana UI
1. Open Kibana: `http://localhost:5601`
2. Click the hamburger menu (☰) on the top-left
3. Navigate to **Analytics** > **Discover**
4. Select "Detections" from the data view dropdown

### What you'll see:
- All detection documents from the ML bot detector
- Real-time data as it streams in
- Fields: `client_ip`, `bot_probability`, `is_bot`, `requests_per_ip`, etc.

---

## 2. 🔧 Filebeat Integration Setup

### What is Filebeat?
Filebeat is a lightweight agent that sends log files to Elasticsearch. It will collect:
- Nginx access logs
- Application logs
- Custom log files

### Step 1: Install Filebeat

#### Option A: Docker Integration (RECOMMENDED)

Add to your `/root/ia/webshield/docker/docker-compose.yml`:

```yaml
  filebeat:
    image: docker.elastic.co/beats/filebeat:8.11.0
    container_name: webshield-filebeat
    user: root
    depends_on:
      elasticsearch:
        condition: service_healthy
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
      KIBANA_HOST: http://kibana:5601
    volumes:
      - ../filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - ../logs/nginx:/var/log/nginx:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: filebeat -e -strict.perms=false
    networks:
      - webshield-network
    restart: unless-stopped
```

#### Option B: Standalone Installation

```bash
# Download Filebeat for your OS
curl -L -O https://artifacts.elastic.co/downloads/beats/filebeat/filebeat-8.11.0-linux-x86_64.tar.gz

# Extract
tar xzf filebeat-8.11.0-linux-x86_64.tar.gz
cd filebeat-8.11.0-linux-x86_64

# Configure (see next section)
# Run
./filebeat -e
```

### Step 2: Create Filebeat Configuration

Create `/root/ia/webshield/filebeat/filebeat.yml`:

```yaml
# Filebeat inputs
filebeat.inputs:
  # Nginx access logs
  - type: log
    enabled: true
    paths:
      - /var/log/nginx/access.log
      - /logs/nginx/*.log
    fields:
      source: nginx
      environment: production
    tags: ["nginx", "webshield"]

  # Application logs (if any)
  - type: log
    enabled: true
    paths:
      - /root/ia/webshield/logs/**/*.log
    fields:
      source: application
      environment: production
    tags: ["webshield", "app"]

# Output to Elasticsearch
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "filebeat-%{[agent.version]}-%%{+yyyy.MM.dd}"

# Kibana integration
kibana:
  host: "kibana:5601"

# Logging
logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

### Step 3: Start Filebeat

```bash
# Using Docker Compose
cd /root/ia/webshield/docker
docker compose up -d filebeat

# Or if installed standalone
/path/to/filebeat -e
```

### Step 4: Verify Filebeat is Working

```bash
# Check container logs
docker logs webshield-filebeat

# Check Elasticsearch for filebeat indices
curl http://localhost:9200/_cat/indices?v | grep filebeat
```

---

## 3. 📊 Accessing Data in Kibana

### View WebBotShield Detections

1. **Go to Discover**
   - URL: `http://localhost:5601/app/discover`
   - Select data view: **"Detections"**

2. **Filter by Bot Activity**
   - KQL Query: `is_bot:1`
   - Shows only detected bot traffic

3. **Search by IP Address**
   - KQL Query: `client_ip:"10.0.10.*"`
   - Shows all requests from specific IP range

4. **High Bot Probability Detections**
   - KQL Query: `bot_probability:[0.7 TO 1]`
   - Shows highly confident bot detections

### View Filebeat Logs

1. **Go to Discover**
   - Select data view: **"filebeat-*"** (once created)

2. **View Nginx Access Logs**
   - Auto-parsed fields: `host`, `remote_ip`, `method`, `request`, `status_code`, etc.

3. **Filter Logs**
   - Failed requests: `response:>=400`
   - Specific path: `request:"/api/login"`
   - Specific user agent: `user_agent:*bot*`

---

## 4. 🎯 Create a Unified Dashboard

Once you have both data sources, create a dashboard showing:
- Bot detection timeline
- Top detected bot IPs
- Nginx error rate over time
- Request distribution by status code
- Top requested paths with bot traffic

### Dashboard Setup (Optional)

See `KIBANA_DASHBOARD_SETUP.md` for automated dashboard creation.

---

## 5. 🔗 Connection Verification Checklist

✅ **Kibana to Elasticsearch**
- [ ] Kibana status shows "Connected"
- [ ] Data views visible in Discover
- [ ] Can query indices

✅ **Filebeat to Elasticsearch**
- [ ] Filebeat logs show "Connected"
- [ ] `filebeat-*` indices appear in Elasticsearch
- [ ] Documents are indexed

✅ **Kibana UI**
- [ ] Can access Discover: `http://localhost:5601/app/discover`
- [ ] Can select "Detections" data view
- [ ] Documents load and display

---

## 6. 🚀 Quick Start Commands

```bash
# Verify Kibana is running
curl -s http://localhost:5601/api/status | grep -q "green" && echo "✅ Kibana OK"

# Verify Elasticsearch connection
curl -s http://localhost:9200/_cluster/health | grep -q "green\|yellow" && echo "✅ ES OK"

# Check available data views
curl -s http://localhost:5601/api/data_views | jq '.data_view[] | {title, name}'

# Check available indices
curl -s http://localhost:9200/_cat/indices?v

# View filebeat setup status (after installation)
docker logs webshield-filebeat | grep -i "setup\|connected\|shipped"
```

---

## 7. 📋 Troubleshooting

### Kibana not connecting to Elasticsearch

**Error**: "Elasticsearch is not available"

**Solution**:
```bash
# Check if Elasticsearch is running
docker ps | grep elasticsearch

# Check connectivity from Kibana
docker exec webshield-kibana curl -s http://elasticsearch:9200/_cluster/health

# Verify network
docker network inspect webshield_default
```

### Filebeat not sending data

**Error**: No `filebeat-*` indices in Elasticsearch

**Solution**:
```bash
# Check Filebeat logs
docker logs webshield-filebeat

# Verify file paths exist
docker exec webshield-filebeat ls -la /var/log/nginx/

# Test Filebeat setup
docker exec webshield-filebeat filebeat setup
```

### Data view not showing documents

**Error**: Empty Discover page

**Solution**:
```bash
# Verify index has data
curl http://localhost:9200/webshield-detections/_count

# Check time range in Kibana
# Set to "Last 1 hour" or "Last 24 hours"

# Check for time field
curl http://localhost:9200/webshield-detections/_mapping | jq '.webshield-detections.mappings.properties | keys'
```

---

## 8. 📚 Additional Resources

- **Kibana Discover Guide**: https://www.elastic.co/guide/en/kibana/8.11/discover.html
- **Filebeat Documentation**: https://www.elastic.co/guide/en/beats/filebeat/8.11/
- **KQL Query Syntax**: https://www.elastic.co/guide/en/kibana/8.11/kuery-query.html
- **Data Views**: https://www.elastic.co/guide/en/kibana/8.11/data-views.html

---

## Next Steps

1. ✅ Data view created for detections
2. 📦 Install Filebeat (Docker compose ready)
3. 📊 Create visualizations in Discover
4. 📈 Build comprehensive dashboard
5. 🔔 Set up alerts (optional)

For questions or issues, check the troubleshooting section above.
