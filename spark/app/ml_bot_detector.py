"""
WebBotShield - ML-Based Real-time Bot Detection with Spark Structured Streaming

This application uses a pre-trained scikit-learn model for bot detection.
It consumes Nginx access logs from Kafka, engineers features matching the training pipeline,
and classifies traffic as Normal or BotAttack using the ML model.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, count, sum as _sum, avg, stddev,
    from_json, to_timestamp, lit, when, udf,
    lower, concat_ws, current_timestamp, first,
    pandas_udf, PandasUDFType, coalesce, struct
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, LongType, FloatType, DoubleType,
    ArrayType
)
import pandas as pd
import numpy as np
import joblib
import yaml
import os


class MLBotDetector:
    """Real-time ML-based bot detection using Spark Structured Streaming"""

    # Feature names matching the trained model
    FEATURE_NAMES = [
        'Port', 'Request_Type', 'Protocol', 'Payload_Size',
        'requests_per_ip', 'failure_rate_per_ip', 'avg_payload_per_ip', 'payload_std_per_ip',
        'time_since_last_request', 'inter_arrival_std',
        'ua_is_empty', 'ua_is_tool', 'ua_is_browser',
        'is_encrypted', 'is_failure', 'protocol_error_rate'
    ]

    # Bot/tool keywords for User-Agent detection
    BOT_KEYWORDS = ['bot', 'crawler', 'spider', 'scan', 'nmap', 'nikto', 'sqlmap', 'curl', 'wget', 'python']
    BROWSER_KEYWORDS = ['mozilla', 'chrome', 'safari', 'firefox', 'edge', 'opera']

    def __init__(self, config_path="../config/settings.yaml", model_path=None):
        """Initialize ML bot detector with configuration and model"""
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()
        
        # Load ML model and scaler
        self.model_path = model_path or self.config.get('ml', {}).get('model_path', '/opt/spark-apps/models/webbotshield_bot_detector.joblib')
        self.scaler_path = self.config.get('ml', {}).get('scaler_path', '/opt/spark-apps/models/scaler.joblib')
        
        self.model = None
        self.scaler = None
        self._load_model()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        default_config = {
            'kafka': {
                'bootstrap_servers': 'kafka:9092',
                'topic': 'nginx-logs',
                'group_id': 'webshield-ml-bot-detector'
            },
            'elasticsearch': {
                'host': 'elasticsearch',
                'port': 9200,
                'logs_index': 'webshield-logs',
                'detections_index': 'webshield-ml-detections'
            },
            'ml': {
                'model_path': '/opt/spark-apps/models/webbotshield_bot_detector.joblib',
                'scaler_path': '/opt/spark-apps/models/scaler.joblib',
                'prediction_threshold': 0.5,
                'window_duration': '60 seconds',
                'slide_duration': '30 seconds'
            },
            'checkpoint': {
                'location': '/opt/spark-apps/.checkpoints/ml'
            }
        }

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    # Deep merge
                    for key, value in loaded_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value

        return default_config

    def _load_model(self):
        """Load pre-trained scikit-learn or ONNX model and scaler"""
        try:
            # Try ONNX format first (environment-independent)
            onnx_model_path = self.model_path.replace('.joblib', '.onnx')
            print(f"🔍 Looking for ONNX model at: {onnx_model_path}")
            if os.path.exists(onnx_model_path):
                print(f"📄 ONNX model file found ({os.path.getsize(onnx_model_path)} bytes)")
                try:
                    import onnxruntime as rt
                    print(f"⏳ Loading ONNX model with ONNXRuntime...")
                    self.model = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
                    self.is_onnx = True
                    print(f"✅ Loaded ONNX ML model from {onnx_model_path}")
                    if os.path.exists(self.scaler_path):
                        try:
                            self.scaler = joblib.load(self.scaler_path)
                            print(f"✅ Loaded scaler from {self.scaler_path}")
                        except Exception as se:
                            print(f"⚠️ Could not load scaler: {se}")
                    return
                except Exception as e:
                    print(f"⚠️ ONNX loading failed: {type(e).__name__}: {e}")
                    print(f"⚠️ Falling back to joblib format...")

            # Fallback to joblib format (not recommended due to environment issues)
            print(f"🔍 Looking for joblib model at: {self.model_path}")
            if os.path.exists(self.model_path):
                print(f"📄 Joblib model file found ({os.path.getsize(self.model_path)} bytes)")
                try:
                    print(f"⏳ Loading joblib model (may have compatibility issues)...")
                    self.model = joblib.load(self.model_path)
                    self.is_onnx = False
                    print(f"✅ Loaded joblib ML model from {self.model_path}")
                except Exception as e:
                    print(f"❌ Error loading joblib model: {type(e).__name__}: {e}")
                    self.model = None
            else:
                print(f"⚠️ No model files found at {self.model_path} or {onnx_model_path}")
                print(f"⚠️ Model not found. Using rule-based fallback.")
                self.model = None

            if self.model is not None and os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    print(f"✅ Loaded scaler from {self.scaler_path}")
                except Exception as e:
                    print(f"⚠️ Could not load scaler: {e}")
        except Exception as e:
            print(f"❌ Unexpected error loading model: {type(e).__name__}: {e}")
            self.model = None

    def _create_spark_session(self):
        """Create and configure Spark session"""
        return (SparkSession.builder
                .appName("WebBotShield-MLBotDetector")
                .master("local[*]")  # Use all available cores locally - no cluster needed
                .config("spark.jars.packages",
                       "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                       "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0")
                .config("spark.sql.streaming.checkpointLocation",
                       self.config['checkpoint']['location'])
                .config("spark.streaming.stopGracefullyOnShutdown", "true")
                .config("spark.sql.shuffle.partitions", "4")
                .getOrCreate())

    def _define_schema(self):
        """Define schema for incoming log messages"""
        return StructType([
            StructField("timestamp", StringType(), True),
            StructField("client_ip", StringType(), True),
            StructField("method", StringType(), True),
            StructField("path", StringType(), True),
            StructField("status", IntegerType(), True),
            StructField("bytes_sent", LongType(), True),
            StructField("user_agent", StringType(), True),
            StructField("referer", StringType(), True),
            StructField("log_type", StringType(), True),
            # New fields from updated log generator
            StructField("protocol", StringType(), True),
            StructField("port", IntegerType(), True),
            StructField("request_type", StringType(), True)
        ])

    def read_kafka_stream(self):
        """Read streaming data from Kafka"""
        kafka_config = self.config['kafka']

        df = (self.spark
              .readStream
              .format("kafka")
              .option("kafka.bootstrap.servers", kafka_config['bootstrap_servers'])
              .option("subscribe", kafka_config['topic'])
              .option("startingOffsets", "earliest")
              .option("failOnDataLoss", "false")
              .load())

        # Parse JSON messages
        schema = self._define_schema()
        parsed_df = (df
                     .selectExpr("CAST(value AS STRING) as json")
                     .select(from_json(col("json"), schema).alias("data"))
                     .select("data.*")
                     .withColumn("timestamp", to_timestamp(col("timestamp"))))

        return parsed_df

    def extract_base_features(self, df):
        """Extract base features from raw log data"""
        
        # Default port/protocol handling (if not in logs, derive from status/path)
        df = df.withColumn(
            "Port",
            when(col("port").isNotNull(), col("port"))
            .when(col("path").startswith("https"), 443)
            .otherwise(80)
        )
        
        df = df.withColumn(
            "Protocol",
            when(col("protocol").isNotNull(), 
                 when(lower(col("protocol")) == "tcp", 0).otherwise(1))
            .otherwise(0)  # Default to TCP (0)
        )
        
        df = df.withColumn(
            "Request_Type",
            when(col("request_type").isNotNull(),
                 when(lower(col("request_type")) == "https", 1).otherwise(0))
            .when(col("Port") == 443, 1)
            .otherwise(0)  # HTTP = 0, HTTPS = 1
        )
        
        # Payload size (bytes_sent as proxy)
        df = df.withColumn("Payload_Size", col("bytes_sent").cast(DoubleType()))
        
        # Is encrypted (port 443 or HTTPS)
        df = df.withColumn(
            "is_encrypted",
            when((col("Port") == 443) | (col("Request_Type") == 1), 1).otherwise(0)
        )
        
        # Is failure (status >= 400)
        df = df.withColumn(
            "is_failure",
            when(col("status") >= 400, 1).otherwise(0)
        )
        
        # User-Agent features
        df = df.withColumn("ua_lower", lower(col("user_agent")))
        
        # Empty UA
        df = df.withColumn(
            "ua_is_empty",
            when((col("user_agent").isNull()) | (col("user_agent") == ""), 1).otherwise(0)
        )
        
        # Bot/tool keywords
        bot_pattern = '|'.join(self.BOT_KEYWORDS)
        df = df.withColumn(
            "ua_is_tool",
            when(col("ua_lower").rlike(bot_pattern), 1).otherwise(0)
        )
        
        # Browser keywords
        browser_pattern = '|'.join(self.BROWSER_KEYWORDS)
        df = df.withColumn(
            "ua_is_browser",
            when(col("ua_lower").rlike(browser_pattern), 1).otherwise(0)
        )
        
        return df.drop("ua_lower")

    def compute_aggregated_features(self, df):
        """Compute windowed aggregation features per IP"""
        ml_config = self.config['ml']
        
        # Apply windowed aggregations per IP
        windowed_df = (df
            .withWatermark("timestamp", "10 minutes")
            .groupBy(
                window(col("timestamp"),
                      ml_config['window_duration'],
                      ml_config['slide_duration']),
                col("client_ip")
            )
            .agg(
                # IP Aggregation features
                count("*").alias("requests_per_ip"),
                avg("is_failure").alias("failure_rate_per_ip"),
                avg("Payload_Size").alias("avg_payload_per_ip"),
                stddev("Payload_Size").alias("payload_std_per_ip"),
                
                # Error features
                avg("is_failure").alias("protocol_error_rate"),
                
                # Keep first values for non-aggregated features
                first("Port").alias("Port"),
                first("Request_Type").alias("Request_Type"),
                first("Protocol").alias("Protocol"),
                first("Payload_Size").alias("Payload_Size"),
                first("is_encrypted").alias("is_encrypted"),
                first("is_failure").alias("is_failure"),
                first("ua_is_empty").alias("ua_is_empty"),
                first("ua_is_tool").alias("ua_is_tool"),
                first("ua_is_browser").alias("ua_is_browser"),
                first("user_agent").alias("user_agent")
            ))
        
        # Fill nulls for stddev (single request = 0 std)
        windowed_df = windowed_df.fillna(0, subset=["payload_std_per_ip"])
        
        # Add placeholder for temporal features (computed in batch context)
        # In streaming, we approximate with window-level stats
        windowed_df = (windowed_df
            .withColumn("time_since_last_request", lit(0.0))
            .withColumn("inter_arrival_std", lit(0.0)))
        
        return windowed_df

    def predict_with_model(self, df):
        """Apply ML model predictions using Pandas UDF"""

        model = self.model
        scaler = self.scaler
        feature_names = self.FEATURE_NAMES
        threshold = self.config['ml'].get('prediction_threshold', 0.5)
        # For ONNX models, pass the ONNX file path, not the joblib path
        is_onnx = hasattr(model, 'run')  # Check if ONNX model
        model_path = self.model_path.replace('.joblib', '.onnx') if is_onnx else self.model_path
        scaler_path = self.scaler_path

        if model is None:
            # Fallback to rule-based detection
            return self._rule_based_detection(df)

        # Broadcast necessary data for distributed prediction
        # For ONNX models, broadcast paths instead of the model object (can't pickle ONNX InferenceSession)
        # For joblib models, broadcast the model itself
        if is_onnx:
            model_path_broadcast = self.spark.sparkContext.broadcast(model_path)
            scaler_path_broadcast = self.spark.sparkContext.broadcast(scaler_path)
            scaler_broadcast = self.spark.sparkContext.broadcast(scaler)
            is_onnx_broadcast = self.spark.sparkContext.broadcast(True)
        else:
            model_broadcast = self.spark.sparkContext.broadcast(model)
            scaler_broadcast = self.spark.sparkContext.broadcast(scaler)
            is_onnx_broadcast = self.spark.sparkContext.broadcast(False)
            model_path_broadcast = None
            scaler_path_broadcast = None

        # Use mapInPandas for distributed prediction
        return self._apply_model_predictions(df, model_broadcast if not is_onnx else None,
                                           model_path_broadcast if is_onnx else None,
                                           scaler_path_broadcast if is_onnx else None,
                                           scaler_broadcast, is_onnx_broadcast, threshold)

    def _apply_model_predictions(self, df, model_broadcast, model_path_broadcast,
                                scaler_path_broadcast, scaler_broadcast, is_onnx_broadcast, threshold):
        """Apply model predictions using foreachBatch for streaming"""

        # Add placeholder columns for schema resolution
        # Use explicit float type for bot_probability to ensure ES maps it correctly
        df = df.withColumn("bot_probability", lit(0.5).cast(DoubleType()))
        df = df.withColumn("is_bot", lit(0).cast(IntegerType()))
        df = df.withColumn("prediction_method", lit("ml_model"))

        # Define the foreachBatch function for actual ML predictions
        def apply_ml_predictions_batch(batch_df, batch_id):
            """Apply ML model predictions to each micro-batch"""
            if batch_df.rdd.isEmpty():
                return

            batch_df.persist()  # Avoid recomputation

            try:
                is_onnx = is_onnx_broadcast.value
                scaler = scaler_broadcast.value

                # Convert to pandas for batch processing
                pdf = batch_df.toPandas()

                # Prepare features
                features_df = pd.DataFrame()
                for feat in self.FEATURE_NAMES:
                    if feat in pdf.columns:
                        features_df[feat] = pdf[feat].fillna(0).astype(float)
                    else:
                        features_df[feat] = 0.0

                # Debug: show feature stats before scaling
                sample_row = features_df.iloc[0] if len(features_df) > 0 else None
                if sample_row is not None and batch_id % 50 == 0:  # Every 50 batches
                    print(f"DEBUG Batch {batch_id}: Sample features: {dict(sample_row[:5])}")

                # Scale numeric features
                numeric_cols = ['Payload_Size', 'avg_payload_per_ip', 'payload_std_per_ip',
                               'time_since_last_request', 'inter_arrival_std']
                if scaler is not None:
                    # Scale all numeric columns at once to match scaler's fit
                    try:
                        cols_to_scale = [col for col in numeric_cols if col in features_df.columns]
                        if cols_to_scale:
                            scaled_values = scaler.transform(features_df[cols_to_scale])
                            features_df[cols_to_scale] = pd.DataFrame(scaled_values, columns=cols_to_scale, index=features_df.index)
                    except Exception as e:
                        print(f"⚠️ Scaling error: {e}")
                else:
                    print(f"⚠️ WARNING: Scaler is None in batch {batch_id}")

                # Load model on executor and predict
                if is_onnx:
                    import onnxruntime as rt
                    model_path = model_path_broadcast.value
                    model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

                    input_name = model.get_inputs()[0].name
                    output_name = model.get_outputs()[0].name
                    X_array = features_df.values.astype(np.float32)
                    onnx_result = model.run([output_name], {input_name: X_array})
                    probabilities = onnx_result[0].flatten()

                    # Normalize if needed
                    if not (probabilities.max() <= 1.0 and probabilities.min() >= 0.0):
                        probabilities = 1 / (1 + np.exp(-probabilities))

                    # Debug: show prediction stats
                    prob_min, prob_max, prob_mean = probabilities.min(), probabilities.max(), probabilities.mean()
                    print(f"✅ Batch {batch_id}: Applied ONNX predictions ({len(pdf)} rows) | "
                          f"Probs: min={prob_min:.4f}, max={prob_max:.4f}, mean={prob_mean:.4f}")
                else:
                    model = model_broadcast.value
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_df.values)[:, 1]
                    else:
                        probabilities = model.predict(features_df.values).astype(float)

                    print(f"✅ Batch {batch_id}: Applied Joblib predictions ({len(pdf)} rows)")

                # Create prediction results
                predictions = (probabilities >= threshold).astype(int)
                # Ensure probabilities are stored as float64
                pdf['bot_probability'] = probabilities.astype(np.float64)
                pdf['is_bot'] = predictions
                pdf['prediction_method'] = 'ml_model'
                # Generate unique detection ID based on index (Elasticsearch can auto-generate IDs, but we'll create them)
                pdf['detection_id'] = pdf.index.astype(str)

                # Debug: verify probabilities are set correctly
                if batch_id % 100 == 0:
                    print(f"DEBUG: Sample probs in pdf: {pdf['bot_probability'].iloc[:3].tolist()}")

                # Convert back to Spark DataFrame with explicit schema for bot_probability
                from pyspark.sql.types import StructField

                # Get schema from pandas DF
                result_df = self.spark.createDataFrame(pdf)
                # Explicitly cast bot_probability to double to ensure it's stored as float
                result_df = result_df.withColumn("bot_probability", col("bot_probability").cast(DoubleType()))

                # Write to Elasticsearch
                result_df.write \
                    .format("org.elasticsearch.spark.sql") \
                    .option("es.nodes", self.config['elasticsearch']['host']) \
                    .option("es.port", str(self.config['elasticsearch']['port'])) \
                    .option("es.mapping.id", "detection_id") \
                    .mode("append") \
                    .save(f"{self.config['elasticsearch'].get('index_prefix', 'webshield')}-detections")

                print(f"✅ Batch {batch_id}: Written {len(pdf)} predictions to Elasticsearch")

            except Exception as e:
                print(f"❌ Error applying ML predictions in batch {batch_id}: {type(e).__name__}: {e}")
            finally:
                batch_df.unpersist()

        # Start the foreachBatch sink for ML predictions
        ml_query = df.writeStream \
            .foreachBatch(apply_ml_predictions_batch) \
            .option("checkpointLocation", f"{self.config['checkpoint']['location']}/ml_predictions") \
            .start()

        print("📊 ML prediction sink started successfully")

        return df

    def _rule_based_detection(self, df):
        """Fallback rule-based detection when model is not available"""
        detection_config = self.config.get('detection', {})
        burst_threshold = detection_config.get('burst_threshold', 10)
        error_threshold = detection_config.get('error_ratio_threshold', 0.5)
        
        # Calculate bot score based on rules
        df = df.withColumn(
            "burst_score",
            when(col("requests_per_ip") >= burst_threshold, 1.0)
            .when(col("requests_per_ip") >= burst_threshold / 2,
                 (col("requests_per_ip") - burst_threshold / 2) / (burst_threshold / 2))
            .otherwise(0.0)
        )
        
        df = df.withColumn(
            "error_score",
            when(col("failure_rate_per_ip") >= error_threshold, 1.0)
            .otherwise(col("failure_rate_per_ip") / error_threshold)
        )
        
        df = df.withColumn(
            "ua_score",
            when(col("ua_is_tool") == 1, 0.8)
            .when(col("ua_is_empty") == 1, 0.6)
            .otherwise(0.0)
        )
        
        # Combined score
        df = df.withColumn(
            "bot_probability",
            (col("burst_score") * 0.4 + col("error_score") * 0.3 + col("ua_score") * 0.3)
        )
        
        df = df.withColumn(
            "is_bot",
            when(col("bot_probability") >= 0.5, 1).otherwise(0)
        )
        
        df = df.withColumn("prediction_method", lit("rule_based"))
        
        return df.drop("burst_score", "error_score", "ua_score")

    def enrich_detections(self, df):
        """Add metadata and detection reasons to results"""
        
        df = df.withColumn("detection_timestamp", current_timestamp())
        
        # Generate detection reasons
        df = df.withColumn(
            "detection_reasons",
            concat_ws(", ",
                when(col("requests_per_ip") > 10, lit("high_request_rate")).otherwise(lit("")),
                when(col("failure_rate_per_ip") > 0.5, lit("high_error_ratio")).otherwise(lit("")),
                when(col("ua_is_tool") == 1, lit("bot_user_agent")).otherwise(lit("")),
                when(col("ua_is_empty") == 1, lit("empty_user_agent")).otherwise(lit("")),
                when(col("ua_is_browser") == 0, lit("non_browser_ua")).otherwise(lit(""))
            )
        )
        
        # Classification label
        df = df.withColumn(
            "classification",
            when(col("is_bot") == 1, lit("BotAttack")).otherwise(lit("Normal"))
        )
        
        return df

    def write_to_console(self, df):
        """Write streaming results to console for debugging"""

        # Ensure all required columns exist with default values if missing
        display_df = df.select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "client_ip",
            "requests_per_ip",
            "failure_rate_per_ip",
            "ua_is_tool",
            "ua_is_browser",
            coalesce(col("bot_probability"), lit(0.0)).alias("bot_probability"),
            coalesce(col("is_bot"), lit(0)).alias("is_bot"),
            coalesce(col("classification"), lit("Unknown")).alias("classification"),
            coalesce(col("prediction_method"), lit("unknown")).alias("prediction_method"),
            coalesce(col("detection_reasons"), lit("")).alias("detection_reasons")
        )

        query = (display_df
                 .writeStream
                 .outputMode("update")
                 .format("console")
                 .option("truncate", False)
                 .option("numRows", 50)
                 .start())

        return query

    def write_to_elasticsearch(self, df, index_name=None):
        """Write streaming results to Elasticsearch"""
        es_config = self.config['elasticsearch']
        index_name = index_name or es_config['detections_index']

        # Flatten window column for Elasticsearch
        df_flat = df.select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "client_ip",
            "requests_per_ip",
            "failure_rate_per_ip",
            "avg_payload_per_ip",
            "bot_probability",
            "is_bot",
            "classification",
            "prediction_method",
            "detection_reasons",
            "detection_timestamp",
            "user_agent"
        )

        query = (df_flat
                 .writeStream
                 .outputMode("update")
                 .format("org.elasticsearch.spark.sql")
                 .option("es.nodes", es_config['host'])
                 .option("es.port", str(es_config['port']))
                 .option("es.resource", index_name)
                 .option("es.mapping.id", "client_ip")
                 .option("checkpointLocation", f"{self.config['checkpoint']['location']}/{index_name}")
                 .start())

        return query

    def run(self):
        """Main execution pipeline"""
        print("=" * 80)
        print("🤖 Starting WebBotShield ML Bot Detector")
        print("=" * 80)
        print(f"📡 Kafka: {self.config['kafka']['bootstrap_servers']}")
        print(f"🔍 Elasticsearch: {self.config['elasticsearch']['host']}:{self.config['elasticsearch']['port']}")
        print(f"🧠 Model: {self.model_path}")
        print(f"📊 Features: {len(self.FEATURE_NAMES)}")
        print("=" * 80)

        # Read stream from Kafka
        print("\n📥 Reading stream from Kafka...")
        raw_logs = self.read_kafka_stream()

        # Extract base features
        print("🔧 Extracting base features...")
        enriched_logs = self.extract_base_features(raw_logs)

        # Compute aggregated features
        print("📈 Computing aggregated features...")
        aggregated = self.compute_aggregated_features(enriched_logs)

        # Apply ML predictions (handled via foreachBatch sink)
        print("🧠 Applying ML model predictions via foreachBatch...")
        predictions = self.predict_with_model(aggregated)

        print("\n" + "=" * 80)
        print("✅ ML Bot detection pipeline started successfully!")
        print("🔍 Monitoring for suspicious traffic patterns...")
        print("📊 Predictions written to Elasticsearch")
        print("=" * 80 + "\n")

        # Wait for ML prediction stream to complete
        # The foreachBatch sink is started inside predict_with_model()
        # So we need to keep the driver alive
        try:
            import time
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n✅ Shutting down gracefully...")
            self.spark.stop()


def main():
    """Entry point for ML bot detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WebBotShield ML Bot Detector")
    parser.add_argument("--config", type=str, default="../config/settings.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model file")
    
    args = parser.parse_args()
    
    detector = MLBotDetector(config_path=args.config, model_path=args.model)
    detector.run()


if __name__ == "__main__":
    main()
