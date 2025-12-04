#!/usr/bin/env python3
"""
Setup Kibana data views and integration for WebBotShield
Creates data views for all Elasticsearch indices and configures Kibana UI
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional

KIBANA_URL = "http://localhost:5601"
ES_URL = "http://localhost:9200"
KIBANA_SPACE = "default"

class KibanaManager:
    def __init__(self, kibana_url=KIBANA_URL, es_url=ES_URL):
        self.kibana_url = kibana_url
        self.es_url = es_url
        self.headers = {
            "kbn-xsrf": "true",
            "Content-Type": "application/json"
        }

    def wait_for_kibana(self, timeout=300):
        """Wait for Kibana to be ready"""
        print("⏳ Waiting for Kibana to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.kibana_url}/api/status", timeout=5)
                if resp.status_code == 200:
                    print("✅ Kibana is ready!")
                    return True
            except:
                pass
            time.sleep(5)
        print("❌ Kibana failed to start")
        return False

    def get_elasticsearch_indices(self) -> List[str]:
        """Get all indices from Elasticsearch"""
        try:
            resp = requests.get(f"{self.es_url}/_cat/indices?format=json")
            if resp.status_code == 200:
                indices = [idx['index'] for idx in resp.json()
                          if not idx['index'].startswith('.')]
                print(f"📊 Found {len(indices)} indices:")
                for idx in indices:
                    print(f"   - {idx}")
                return indices
        except Exception as e:
            print(f"❌ Error getting indices: {e}")
        return []

    def check_index_exists(self, index_name: str) -> bool:
        """Check if index exists and has data"""
        try:
            resp = requests.get(f"{self.es_url}/{index_name}/_count")
            if resp.status_code == 200:
                count = resp.json().get('count', 0)
                print(f"   └─ Contains {count} documents")
                return count > 0
        except:
            pass
        return False

    def create_data_view(self, name: str, index_pattern: str,
                        time_field: Optional[str] = None) -> Optional[Dict]:
        """Create a data view (Kibana 8.x) or index pattern (legacy)"""
        print(f"\n📌 Creating data view: {name}")
        print(f"   Index pattern: {index_pattern}")

        # Try modern data view API (Kibana 8.x)
        url = f"{self.kibana_url}/api/data_views/data_view"
        payload = {
            "data_view": {
                "title": index_pattern,
                "name": name,
            }
        }

        if time_field:
            payload["data_view"]["timeFieldName"] = time_field
            print(f"   Time field: {time_field}")

        resp = requests.post(url, headers=self.headers, json=payload)

        if resp.status_code in [200, 201]:
            result = resp.json()
            dv_id = result.get('data_view', {}).get('id')
            print(f"✅ Data view created successfully (ID: {dv_id})")
            return result

        # Fallback to index pattern API (legacy)
        print("   ℹ️ Using legacy index pattern API...")
        url = f"{self.kibana_url}/api/saved_objects/index-pattern"
        payload = {
            "attributes": {
                "title": index_pattern,
                "name": name
            }
        }

        if time_field:
            payload["attributes"]["timeFieldName"] = time_field

        resp = requests.post(url, headers=self.headers, json=payload)

        if resp.status_code in [200, 201]:
            result = resp.json()
            pattern_id = result.get('id')
            print(f"✅ Index pattern created successfully (ID: {pattern_id})")
            return result
        else:
            print(f"❌ Failed to create data view: {resp.status_code}")
            print(f"   Response: {resp.text[:200]}")
            return None

    def detect_time_field(self, index_name: str) -> Optional[str]:
        """Detect the time field in an index"""
        try:
            resp = requests.get(f"{self.es_url}/{index_name}/_mapping")
            if resp.status_code == 200:
                mappings = resp.json().get(index_name, {}).get('mappings', {}).get('properties', {})

                # Look for common time field names
                for field_name in ['@timestamp', 'timestamp', 'detection_timestamp',
                                  'time', 'date', 'event_time', 'log_time']:
                    if field_name in mappings:
                        field_type = mappings[field_name].get('type')
                        if field_type in ['date', 'long']:
                            print(f"   🕐 Detected time field: {field_name} ({field_type})")
                            return field_name
        except:
            pass
        return None

    def setup_webshield_data_views(self):
        """Setup data views for WebBotShield indices"""
        print("=" * 70)
        print("🔧 WebBotShield - Kibana Data Views Setup")
        print("=" * 70)

        # Get all indices
        indices = self.get_elasticsearch_indices()
        if not indices:
            print("❌ No indices found in Elasticsearch")
            return False

        # Create data views for each index
        print("\n" + "=" * 70)
        print("📊 Creating Data Views")
        print("=" * 70)

        created_count = 0
        for index_name in indices:
            print(f"\n🔍 Processing: {index_name}")

            if not self.check_index_exists(index_name):
                print(f"   ⚠️  Index has no data, skipping...")
                continue

            # Detect time field
            time_field = self.detect_time_field(index_name)

            # Create friendly name
            friendly_name = index_name.replace('webshield-', '').replace('-', ' ').title()

            # Create data view
            result = self.create_data_view(friendly_name, f"{index_name}*", time_field)
            if result:
                created_count += 1

        if created_count > 0:
            print("\n" + "=" * 70)
            print(f"✅ Successfully created {created_count} data view(s)")
            print("=" * 70)
            print("\n📍 Access in Kibana:")
            print(f"   1. Go to: {self.kibana_url}/app/kibana")
            print(f"   2. Click: Analytics > Discover")
            print(f"   3. Select your data view to explore data")
            print("\n🔗 Integration Features:")
            print("   - Auto-detection of fields")
            print("   - Time-based filtering")
            print("   - Full-text search")
            print("   - Field exploration")
            return True
        else:
            print("❌ Failed to create any data views")
            return False

    def configure_kibana_ui(self):
        """Configure Kibana UI defaults and settings"""
        print("\n" + "=" * 70)
        print("⚙️ Configuring Kibana UI")
        print("=" * 70)

        # Set default index pattern if possible
        try:
            # Get list of saved objects to find index pattern ID
            resp = requests.get(
                f"{self.kibana_url}/api/saved_objects/index-pattern",
                headers=self.headers
            )
            if resp.status_code == 200:
                saved_objects = resp.json().get('saved_objects', [])
                if saved_objects:
                    # Set first one as default (optional)
                    default_id = saved_objects[0]['id']
                    print(f"✅ Found index patterns")
                    for obj in saved_objects:
                        print(f"   - {obj.get('attributes', {}).get('title')}")
        except Exception as e:
            print(f"ℹ️ Could not set default pattern: {e}")

        return True

    def verify_elasticsearch_connection(self) -> bool:
        """Verify Kibana can reach Elasticsearch"""
        print("\n" + "=" * 70)
        print("🔗 Verifying Elasticsearch Connection")
        print("=" * 70)

        try:
            # Check Elasticsearch health
            resp = requests.get(f"{self.es_url}/_cluster/health")
            if resp.status_code == 200:
                health = resp.json()
                print(f"✅ Elasticsearch is connected")
                print(f"   Status: {health.get('status')}")
                print(f"   Nodes: {health.get('number_of_nodes')}")
                print(f"   Active shards: {health.get('active_shards')}")
                return True
        except Exception as e:
            print(f"❌ Failed to connect to Elasticsearch: {e}")
            return False

    def show_analytics_instructions(self):
        """Show how to access Analytics/Discover in Kibana"""
        print("\n" + "=" * 70)
        print("📈 Analytics & Discovery Guide")
        print("=" * 70)
        print("\n🎯 Access Discover:")
        print(f"   URL: {self.kibana_url}/app/discover")
        print("\n📊 Available Actions:")
        print("   1. View all documents in real-time")
        print("   2. Filter by time range")
        print("   3. Search with KQL (Kibana Query Language)")
        print("   4. Analyze field values and distributions")
        print("   5. Create visualizations from data")
        print("\n🔍 Example Queries:")
        print('   - is_bot:1')
        print('   - bot_probability:[0.7 TO 1]')
        print('   - client_ip:"10.0.10.*"')
        print("=" * 70)

def main():
    if not sys.stdout.isatty():
        # Running in non-interactive mode
        pass

    manager = KibanaManager()

    # Step 1: Wait for Kibana
    if not manager.wait_for_kibana():
        return False

    # Step 2: Verify Elasticsearch connection
    if not manager.verify_elasticsearch_connection():
        print("\n⚠️  Warning: Could not verify Elasticsearch connection")
        print("    Continuing anyway, but Kibana may not be able to connect...")

    # Step 3: Create data views
    if not manager.setup_webshield_data_views():
        return False

    # Step 4: Configure UI
    manager.configure_kibana_ui()

    # Step 5: Show instructions
    manager.show_analytics_instructions()

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
