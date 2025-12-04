# WebBotShield Kibana Dashboard - Complete Guide

## 🎯 Quick Access

**Dashboard URL**: `http://localhost:5601/app/dashboards/view/webshield-main-dashboard`

**Status**: ✅ **LIVE AND ACTIVE**

---

## 📊 Dashboard Overview

The WebBotShield dashboard provides real-time monitoring of bot detection activities with **10 comprehensive visualizations** showing:

- Real-time detection trends
- Bot vs normal traffic classification
- Malicious IP identification
- User agent analysis
- Traffic encryption patterns
- Request success/failure metrics
- Protocol distribution

---

## 📈 Dashboard Visualizations

### 1. **Bot Detection Timeline** 📈
- **Type**: Line chart
- **Shows**: Number of detections over time (60 second intervals)
- **Use**: Identify attack patterns and peak detection times
- **Insight**: Sudden spikes indicate coordinated bot activity

### 2. **Total Detections** 🔢
- **Type**: Metric card (KPI)
- **Shows**: Single large number - total detection records
- **Use**: At-a-glance detection volume
- **Insight**: Quick status indicator for overall threat level

### 3. **Bot vs Normal Classification** 🍰
- **Type**: Pie chart
- **Shows**: Ratio of bot traffic (is_bot=1) to normal traffic (is_bot=0)
- **Use**: Understand threat prevalence
- **Insight**: If heavily tilted toward bots, indicates active attack

### 4. **Top Source IPs by Request Count** 📊
- **Type**: Horizontal bar chart
- **Shows**: Top 10 source IPs making requests (sorted by count)
- **Use**: Identify most active clients and suspicious origins
- **Insight**: High-frequency IPs from unusual locations should be investigated

### 5. **Bot Tools Detection** 🤖
- **Type**: Bar chart
- **Shows**: Count of requests with tool user agents (ua_is_tool) vs regular UAs
- **Use**: Detect automated tooling usage
- **Insight**: High tool counts = suspicious automated activity

### 6. **Encrypted vs Unencrypted Traffic** 🔐
- **Type**: Pie chart
- **Shows**: HTTPS vs HTTP traffic split (is_encrypted)
- **Use**: Understand protocol security posture
- **Insight**: Bots often use unencrypted HTTP; legitimate users prefer HTTPS

### 7. **Failed vs Successful Requests** ✅
- **Type**: Pie chart
- **Shows**: Success rate of requests (is_failure: 0=success, 1=failure)
- **Use**: Identify probing/scanning attempts
- **Insight**: High failure rates indicate reconnaissance or fuzzing attacks

### 8. **Browser Traffic** 🌐
- **Type**: Bar chart
- **Shows**: Split between browser user agents (ua_is_browser) and non-browser UAs
- **Use**: Distinguish real user traffic from automated requests
- **Insight**: Low browser count may indicate predominant bot activity

### 9. **Request Frequency Over Time** 📅
- **Type**: Line chart
- **Shows**: Request volume trends across time
- **Use**: Detect traffic anomalies and patterns
- **Insight**: Abrupt changes or patterns indicate coordinated attacks

### 10. **Traffic by Port** 🔌
- **Type**: Pie chart
- **Shows**: Distribution of traffic across ports (Port field)
- **Use**: Understand target protocols and services
- **Insight**: Unusual ports or high traffic on non-standard ports = suspicious

---

## 🚀 How to Use the Dashboard

### Viewing Data
```
1. Open: http://localhost:5601/app/dashboards/view/webshield-main-dashboard
2. Dashboard auto-refreshes every 10 seconds
3. Adjust time range using picker at top (default: last 7 days)
```

### Filtering & Drilling Down
```
Click any visualization to:
  • Filter other panels to show only that data
  • Example: Click "1" in "Bot vs Normal" pie → see only bot detections

Click visualization title to:
  • See detailed raw data in table format
  • Download the data

Click panel ⋮ menu to:
  • Inspect aggregation details
  • Customize visualization
  • View raw query
```

### Customizing the Dashboard
```
1. Click "Edit" button (top right)
2. Drag panels to rearrange layout
3. Click panel menu (⋮) to:
   - View raw data
   - Export data
   - Customize colors/fields
   - Delete panel
4. Click "Add panel" to add more visualizations
5. Click "Save" when done
```

---

## 📊 Key Data Fields

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `bot_probability` | Float | ML model confidence score | 0.0-1.0 |
| `is_bot` | Integer | Bot classification result | 0=normal, 1=bot |
| `client_ip` | Text | Source IP address | IP format (xxx.xxx.xxx.xxx) |
| `requests_per_ip` | Integer | Requests in 60s window | 1-N |
| `failure_rate_per_ip` | Float | Error ratio for IP | 0.0-1.0 |
| `avg_payload_per_ip` | Float | Average request size | bytes |
| `payload_std_per_ip` | Float | Payload size variance | bytes² |
| `ua_is_empty` | Integer | Empty user agent | 0=has UA, 1=empty |
| `ua_is_tool` | Integer | Bot/tool user agent | 0=other, 1=tool |
| `ua_is_browser` | Integer | Browser user agent | 0=other, 1=browser |
| `is_encrypted` | Integer | HTTPS/TLS usage | 0=HTTP, 1=HTTPS |
| `is_failure` | Integer | Request outcome | 0=success, 1=failure |
| `Port` | Integer | Target port | 80, 443, etc. |
| `Protocol` | Integer | Protocol type | 0=TCP, 1=other |
| `window.start` | Timestamp | Detection window start | milliseconds epoch |
| `window.end` | Timestamp | Detection window end | milliseconds epoch |
| `user_agent` | Text | Full user agent string | Browser/bot identifier |

---

## 🎯 Common Investigation Workflows

### 🚨 Detecting Active Bot Attacks
```
1. Look at "Bot Detection Timeline" for spikes
2. Check "Bot vs Normal Classification" ratio
3. Click top ip in "Top Source IPs"
4. Review "Bot Tools Detection" for automated activity
5. Go to Discover → search: is_bot:1 AND bot_probability:[0.8 TO 1]
```

### 🔍 Investigating a Specific IP
```
1. Go to Discover tab
2. Search: client_ip:"<IP_ADDRESS>"
3. Review that IP's:
   - Request frequency (requests_per_ip)
   - Success/failure rate (is_failure)
   - User agent patterns (ua_is_tool, ua_is_browser)
   - Payload sizes (avg_payload_per_ip)
   - Encryption usage (is_encrypted)
```

### 📈 Performance Analysis
```
1. Check "Request Frequency Over Time" for traffic patterns
2. Monitor "Traffic by Port" for unexpected protocol usage
3. Review "Failed vs Successful" for probing detection
4. Analyze "Encrypted vs Unencrypted" for protocol preference
```

### 🔐 Security Posture Check
```
1. Monitor bot detection ratio (should be LOW for normal traffic)
2. Check "Bot Tools Detection" for crawler activity
3. Review "Encrypted vs Unencrypted" (prefer HTTPS)
4. Monitor "Failed vs Successful" (spikes = scanning)
5. Inspect "Top Source IPs" for known malicious ranges
```

---

## 💾 Export & Reporting

### Export Dashboard Data
```
1. Click any visualization ⋮ menu
2. Select "Download" → choose format:
   - CSV (for spreadsheets)
   - JSON (for analysis)
3. Use for reports or external analysis
```

### Create Custom Reports
```
1. Use Kibana's reporting feature:
   - Click "Share" → "Generate PDF/PNG"
2. Or save filtered views for scheduled reports
```

---

## 🔧 Advanced Features

### Save Filtered Views
```
1. Apply filters in Discover
2. Click "Save as" to create saved search
3. Add to dashboard as new panel
```

### Create Alerts
```
1. Click dashboard menu (⋮) → "Create alert"
2. Set condition: bot_probability > 0.8
3. Configure notification (email, Slack, webhook)
4. Set check frequency (every 5 min, hourly, etc.)
```

### Share Dashboard with Team
```
1. Click "Share" button
2. Copy URL to share with colleagues
3. Or generate embed code for security portal
4. Set permissions (view-only or edit)
```

---

## 🚨 Troubleshooting

### Dashboard shows no data
- ✓ Verify time range includes current data
- ✓ Check ML detector is running: `docker ps | grep ml-bot`
- ✓ Go to Discover to verify data exists
- ✓ Run: `curl -s http://localhost:9200/webshield-detections/_count`

### Visualizations won't load
- ✓ Try refreshing browser (Ctrl+F5)
- ✓ Check Kibana logs: `docker logs webshield-kibana | tail -100`
- ✓ Verify Elasticsearch: `curl http://localhost:9200/_cluster/health`

### Data looks outdated
- ✓ Check auto-refresh is enabled (should be 10s)
- ✓ Verify ML detector: `docker logs webshield-spark-ml-bot-detector | grep "Batch"`
- ✓ Check Elasticsearch writes: `curl -s http://localhost:9200/webshield-detections/_search?size=1`

---

## 📝 Dashboard Refresh Schedule

- **Auto-refresh**: Every 10 seconds
- **Time range**: Last 7 days (configurable at top)
- **Data source**: `webshield-detections` index in Elasticsearch
- **Update latency**: Real-time (streaming from ML detector)

---

## 🔗 Related Documentation

- [Kibana Integration Guide](./KIBANA_INTEGRATION_GUIDE.md)
- [ML Bot Detector Setup](../README.md)
- [Elasticsearch Configuration](../../docker/docker-compose.yml)
- [Kibana API Docs](http://localhost:5601/app/dev_tools#/console)

---

## 📊 Example Queries for Discover

### Find All Bot Detections
```
is_bot:1
```

### High Confidence Bot Detections
```
bot_probability:[0.8 TO 1]
```

### Specific IP Analysis
```
client_ip:"10.0.10.32"
```

### Tool-Based Attacks
```
ua_is_tool:1 AND is_bot:1
```

### Failed Requests (Scanning)
```
is_failure:1 AND requests_per_ip:[5 TO 100]
```

### HTTPS Only Traffic
```
is_encrypted:1
```

### High Request Rate IPs
```
requests_per_ip:[10 TO 1000]
```

### Combined Analysis
```
(is_bot:1 OR ua_is_tool:1) AND failure_rate_per_ip:[0.5 TO 1]
```

---

## ✅ Dashboard Status

| Component | Status |
|-----------|--------|
| Data Source | ✅ Active (`webshield-detections`) |
| Visualizations | ✅ 10 charts created |
| Auto-refresh | ✅ Enabled (10 seconds) |
| Data Latency | ✅ Real-time (<1 second) |
| Elasticsearch | ✅ Connected |
| Kibana | ✅ Running (port 5601) |

---

**Last Updated**: 2025-12-03
**Dashboard Name**: WebBotShield - Bot Detection Dashboard
**Auto-refresh**: Every 10 seconds
**Data Updates**: Real-time from ML detector
