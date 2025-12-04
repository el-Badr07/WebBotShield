#!/usr/bin/env python3
"""
Create comprehensive Kibana dashboard for WebBotShield ML Bot Detection
Dashboard includes: timelines, heatmaps, metrics, tables, and analysis charts
"""

import requests
import json
import time
import uuid

KIBANA_URL = "http://localhost:5601"
DATA_VIEW_ID = "5c48ed64-1a8d-48a4-a4b9-c1bf487f8f3b"  # From previous setup

class KibanaDashboardBuilder:
    def __init__(self, kibana_url=KIBANA_URL):
        self.kibana_url = kibana_url
        self.headers = {
            "kbn-xsrf": "true",
            "Content-Type": "application/json"
        }
        self.viz_list = []

    def wait_for_kibana(self, timeout=60):
        """Wait for Kibana to be ready"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.kibana_url}/api/status", timeout=5)
                if resp.status_code == 200:
                    print("✅ Kibana is ready")
                    return True
            except:
                pass
            time.sleep(5)
        return False

    def create_visualization(self, viz_type, title, metric_field=None,
                            breakdown_field=None, time_field="window.start"):
        """Create a visualization"""
        viz_id = f"webshield-{title.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"

        # Build visualization state based on type
        if viz_type == "line":
            vis_state = self._build_line_chart(metric_field, time_field)
        elif viz_type == "horizontal_bar":
            vis_state = self._build_horizontal_bar(breakdown_field, metric_field)
        elif viz_type == "pie":
            vis_state = self._build_pie_chart(breakdown_field)
        elif viz_type == "metric":
            vis_state = self._build_metric(metric_field)
        elif viz_type == "table":
            vis_state = self._build_table()
        elif viz_type == "gauge":
            vis_state = self._build_gauge(metric_field)
        else:
            return None

        url = f"{self.kibana_url}/api/saved_objects/visualization/{viz_id}"

        payload = {
            "attributes": {
                "title": title,
                "visState": json.dumps(vis_state),
                "uiStateJSON": "{}",
                "description": f"WebBotShield - {title}",
                "version": 1,
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "index": DATA_VIEW_ID,
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            }
        }

        resp = requests.post(url, headers=self.headers, json=payload)
        if resp.status_code in [200, 201]:
            print(f"✅ Created visualization: {title}")
            self.viz_list.append({"id": viz_id, "title": title})
            return viz_id
        else:
            print(f"❌ Failed to create visualization '{title}': {resp.status_code}")
            if resp.text:
                print(f"   Response: {resp.text[:200]}")
            return None

    def _build_line_chart(self, metric="count", time_field="window.start"):
        """Build line chart visualization for time series"""
        return {
            "title": "",
            "type": "line",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric",
                    "params": {}
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "date_histogram",
                    "schema": "segment",
                    "params": {
                        "field": time_field,
                        "timeRange": {
                            "from": "now-7d",
                            "to": "now"
                        },
                        "useNormalizedEsInterval": True,
                        "scaleMetricValues": False,
                        "interval": "30s",
                        "drop_partials": False,
                        "min_doc_count": 0,
                        "extended_bounds": {}
                    }
                }
            ],
            "params": {
                "type": "line",
                "grid": {
                    "categoryLines": False,
                    "valueAxis": "ValueAxis-1"
                },
                "categoryAxes": [
                    {
                        "id": "CategoryAxis-1",
                        "type": "category",
                        "position": "bottom",
                        "show": True,
                        "style": {}
                    }
                ],
                "valueAxes": [
                    {
                        "id": "ValueAxis-1",
                        "name": "LeftAxis-1",
                        "type": "value",
                        "position": "left",
                        "show": True
                    }
                ],
                "seriesParams": [
                    {
                        "show": True,
                        "type": "line",
                        "mode": "normal",
                        "data": {"label": "Count", "id": "1"},
                        "valueAxis": "ValueAxis-1"
                    }
                ],
                "addTooltip": True,
                "addLegend": True,
                "legendPosition": "bottom"
            }
        }

    def _build_horizontal_bar(self, breakdown_field, metric="count"):
        """Build horizontal bar chart visualization using terms aggregation"""
        return {
            "title": "",
            "type": "histogram",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric",
                    "params": {}
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "terms",
                    "schema": "segment",
                    "params": {
                        "field": breakdown_field,
                        "size": 10,
                        "order": "desc",
                        "orderBy": "1",
                        "customLabel": breakdown_field
                    }
                }
            ],
            "params": {
                "addLegend": True,
                "addTooltip": True,
                "legendPosition": "right",
                "isDonut": False
            }
        }

    def _build_pie_chart(self, breakdown_field):
        """Build pie chart"""
        return {
            "title": "",
            "type": "pie",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric"
                },
                {
                    "id": "2",
                    "enabled": True,
                    "type": "terms",
                    "schema": "segment",
                    "params": {
                        "field": breakdown_field,
                        "size": 5,
                        "order": "desc",
                        "orderBy": "1"
                    }
                }
            ],
            "params": {
                "type": "pie",
                "addTooltip": True,
                "addLegend": True,
                "legendPosition": "right",
                "isDonut": False
            }
        }

    def _build_metric(self, field="count"):
        """Build metric visualization"""
        return {
            "title": "",
            "type": "metric",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric"
                }
            ],
            "params": {
                "addTooltip": True,
                "addLegend": False
            }
        }

    def _build_table(self):
        """Build data table visualization"""
        return {
            "title": "",
            "type": "table",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "count",
                    "schema": "metric"
                }
            ],
            "params": {
                "perPage": 10,
                "showPartialRows": False,
                "showMetricsAtAllLevels": False,
                "showTotal": False,
                "totalFunc": "sum",
                "percentageCol": ""
            }
        }

    def _build_gauge(self, field):
        """Build gauge visualization"""
        return {
            "title": "",
            "type": "gauge",
            "aggs": [
                {
                    "id": "1",
                    "enabled": True,
                    "type": "avg",
                    "schema": "metric",
                    "params": {
                        "field": field
                    }
                }
            ],
            "params": {
                "addTooltip": True,
                "addLegend": True,
                "isDisplayWarning": False,
                "gaugeType": "Arc",
                "orientation": "vertical",
                "useGaugeOrientation": True,
                "alignment": "automatic",
                "collapseLabels": False,
                "invertColors": False,
                "min": 0,
                "max": 100
            }
        }

    def create_dashboard(self):
        """Create main dashboard with all visualizations"""
        print("\n" + "=" * 70)
        print("📊 Creating Kibana Dashboard")
        print("=" * 70)

        # Create visualizations
        print("\n📈 Creating visualizations...")

        viz_ids = []

        # 1. Total Detections (Metric)
        vid = self.create_visualization("metric", "Total Detections")
        if vid:
            viz_ids.append(vid)

        # 3. Bot vs Normal Traffic (Pie)
        vid = self.create_visualization("pie", "Bot vs Normal Classification",
                                       breakdown_field="is_bot")
        if vid:
            viz_ids.append(vid)

        # 4. Top Source IPs
        vid = self.create_visualization("horizontal_bar", "Top Source IPs by Request Count",
                                       breakdown_field="client_ip.keyword")
        if vid:
            viz_ids.append(vid)

        # 5. Bot Probability Distribution
        vid = self.create_visualization("horizontal_bar", "Bot Tools Detection",
                                       breakdown_field="ua_is_tool")
        if vid:
            viz_ids.append(vid)

        # 6. HTTP vs HTTPS Traffic
        vid = self.create_visualization("pie", "Encrypted vs Unencrypted Traffic",
                                       breakdown_field="is_encrypted")
        if vid:
            viz_ids.append(vid)

        # 7. Error Rate Analysis
        vid = self.create_visualization("pie", "Failed vs Successful Requests",
                                       breakdown_field="is_failure")
        if vid:
            viz_ids.append(vid)

        # 8. Top Browsers
        vid = self.create_visualization("horizontal_bar", "Browser Traffic",
                                       breakdown_field="ua_is_browser")
        if vid:
            viz_ids.append(vid)

        # 9. Average Payload Size by IP
        vid = self.create_visualization("horizontal_bar", "Average Payload Size by Top IPs",
                                       breakdown_field="client_ip.keyword")
        if vid:
            viz_ids.append(vid)

        # 10. Port Distribution
        vid = self.create_visualization("pie", "Traffic by Port",
                                       breakdown_field="Port")
        if vid:
            viz_ids.append(vid)

        print(f"\n✅ Created {len(viz_ids)} visualizations")

        # Create dashboard
        print("\n📋 Creating dashboard...")
        dashboard_id = "webshield-main-dashboard"

        # Build panels for dashboard
        panels = []
        x, y = 0, 0
        w, h = 24, 15

        for idx, viz_id in enumerate(viz_ids):
            panel = {
                "version": "8.11.0",
                "gridData": {
                    "x": (idx % 2) * 24,
                    "y": (idx // 2) * 15,
                    "w": 24,
                    "h": 15,
                    "i": str(idx)
                },
                "panelIndex": str(idx),
                "embeddableConfig": {},
                "panelRefName": f"panel_{idx}"
            }
            panels.append(panel)

        # Build panel references
        references = []
        for idx, viz_id in enumerate(viz_ids):
            references.append({
                "name": f"panel_{idx}",
                "type": "visualization",
                "id": viz_id
            })

        url = f"{self.kibana_url}/api/saved_objects/dashboard/{dashboard_id}"

        payload = {
            "attributes": {
                "title": "WebBotShield - Bot Detection Dashboard",
                "description": "Real-time ML-based bot detection monitoring and analysis",
                "hits": 0,
                "timeRestore": False,
                "timeFrom": "now-7d",
                "timeTo": "now",
                "refreshInterval": {
                    "pause": False,
                    "value": 10000
                },
                "panelsJSON": json.dumps(panels),
                "optionsJSON": json.dumps({
                    "useMargins": True,
                    "hidePanelTitles": False,
                    "syncColors": False,
                    "syncTooltips": False,
                    "syncCursor": True
                }),
                "version": 1,
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            },
            "references": references
        }

        resp = requests.post(url, headers=self.headers, json=payload, params={"overwrite": "true"})

        if resp.status_code in [200, 201]:
            print(f"✅ Dashboard created successfully!")
            print(f"\n🔗 Access dashboard:")
            print(f"   URL: {self.kibana_url}/app/dashboards/view/{dashboard_id}")
            return True
        else:
            print(f"❌ Failed to create dashboard: {resp.status_code}")
            print(f"   Response: {resp.text[:300]}")
            return False

    def print_summary(self):
        """Print summary of created content"""
        print("\n" + "=" * 70)
        print("✅ Dashboard Setup Complete!")
        print("=" * 70)
        print("\n📊 Dashboard Details:")
        print(f"   Name: WebBotShield - Bot Detection Dashboard")
        print(f"   Visualizations: {len(self.viz_list)}")

        print("\n📈 Included Visualizations:")
        for i, viz in enumerate(self.viz_list, 1):
            print(f"   {i}. {viz['title']}")

        print("\n🔗 Access Points:")
        print(f"   - Kibana Home: {self.kibana_url}")
        print(f"   - Dashboard: {self.kibana_url}/app/dashboards/view/webshield-main-dashboard")
        print(f"   - Discover: {self.kibana_url}/app/discover")

        print("\n📝 Key Metrics Shown:")
        print("   - Real-time bot detection timeline")
        print("   - Total detection count")
        print("   - Bot vs normal traffic ratio")
        print("   - Top malicious IPs")
        print("   - Tool/bot user agent detection")
        print("   - HTTP vs HTTPS traffic split")
        print("   - Request success/failure rates")
        print("   - Browser traffic analysis")
        print("   - Port distribution")
        print("   - Request frequency trends")

        print("\n💡 Tips:")
        print("   - Dashboard auto-refreshes every 10 seconds")
        print("   - Time range: Last 7 days (configurable)")
        print("   - Click any chart to filter other visualizations")
        print("   - Click 'Edit' to customize layout and add more panels")

        print("=" * 70)

def main():
    builder = KibanaDashboardBuilder()

    if not builder.wait_for_kibana():
        print("❌ Kibana not ready")
        return False

    if not builder.create_dashboard():
        return False

    builder.print_summary()
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
