"""
WebBotShield - Real-time Bot Detection with Spark Structured Streaming

This application consumes Nginx access logs from Kafka, analyzes traffic patterns,
and detects bot behavior using temporal analysis, error ratios, and User-Agent patterns.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, count, sum as _sum, avg,
    from_json, to_timestamp, lit, when,
    regexp_extract, lower, concat_ws, current_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, LongType, TimestampType
)
import yaml
import os


class BotDetector:
    """Real-time bot detection using Spark Structured Streaming"""

    def __init__(self, config_path="../config/settings.yaml"):
        """Initialize bot detector with configuration"""
        self.config = self._load_config(config_path)
        self.spark = self._create_spark_session()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        default_config = {
            'kafka': {
                'bootstrap_servers': 'kafka:9092',
                'topic': 'nginx-logs',
                'group_id': 'webshield-bot-detector'
            },
            'elasticsearch': {
                'host': 'elasticsearch',
                'port': 9200,
                'logs_index': 'webshield-logs',
                'detections_index': 'webshield-detections'
            },
            'detection': {
                'burst_threshold': 10,  # requests per minute (more sensitive)
                'error_ratio_threshold': 0.5,  # 50% errors
                'window_duration': '30 seconds',  # Shorter window for more sensitive detection
                'slide_duration': '10 seconds',  # More frequent evaluation
                'bot_score_threshold': 0.5  # Lower threshold for detection
            },
            'checkpoint': {
                'location': '/opt/spark-apps/.checkpoints'
            }
        }

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)

        return default_config

    def _create_spark_session(self):
        """Create and configure Spark session"""
        return (SparkSession.builder
                .appName("WebBotShield-BotDetector")
                .config("spark.jars.packages",
                       "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                       "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0")
                .config("spark.sql.streaming.checkpointLocation",
                       self.config['checkpoint']['location'])
                .config("spark.streaming.stopGracefullyOnShutdown", "true")
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
            StructField("log_type", StringType(), True)
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

    def detect_suspicious_user_agents(self, df):
        """Detect suspicious User-Agent patterns"""
        # Common bot patterns
        bot_patterns = [
            'bot', 'crawler', 'spider', 'scraper', 'curl', 'wget',
            'python-requests', 'scrapy', 'java', 'go-http-client',
            'apache-httpclient', 'libwww', 'mechanize'
        ]

        pattern = '|'.join(bot_patterns)

        df = df.withColumn(
            "ua_suspicious",
            when(
                (col("user_agent").isNull()) |
                (col("user_agent") == "") |
                (lower(col("user_agent")).rlike(pattern)),
                1
            ).otherwise(0)
        )

        return df

    def detect_error_patterns(self, df):
        """Categorize HTTP status codes"""
        df = (df
              .withColumn("is_error_4xx", when((col("status") >= 400) & (col("status") < 500), 1).otherwise(0))
              .withColumn("is_error_5xx", when((col("status") >= 500) & (col("status") < 600), 1).otherwise(0))
              .withColumn("is_error", when((col("status") >= 400), 1).otherwise(0))
              .withColumn("is_success", when((col("status") >= 200) & (col("status") < 300), 1).otherwise(0)))

        return df

    def analyze_traffic_patterns(self, df):
        """Analyze traffic patterns using windowed aggregations"""
        detection_config = self.config['detection']

        # Apply windowed aggregations per IP
        windowed_df = (df
                       .withWatermark("timestamp", "10 minutes")
                       .groupBy(
                           window(col("timestamp"),
                                 detection_config['window_duration'],
                                 detection_config['slide_duration']),
                           col("client_ip")
                       )
                       .agg(
                           count("*").alias("request_count"),
                           _sum("is_error_4xx").alias("error_4xx_count"),
                           _sum("is_error_5xx").alias("error_5xx_count"),
                           _sum("is_error").alias("total_errors"),
                           _sum("is_success").alias("total_success"),
                           _sum("ua_suspicious").alias("suspicious_ua_count"),
                           avg("bytes_sent").alias("avg_bytes_sent")
                       ))

        # Calculate metrics
        windowed_df = (windowed_df
                       .withColumn("error_ratio",
                                  when(col("request_count") > 0,
                                       col("total_errors") / col("request_count"))
                                  .otherwise(0))
                       .withColumn("ua_suspicious_ratio",
                                  when(col("request_count") > 0,
                                       col("suspicious_ua_count") / col("request_count"))
                                  .otherwise(0)))

        return windowed_df

    def calculate_bot_score(self, df):
        """Calculate bot likelihood score based on multiple factors"""
        detection_config = self.config['detection']
        burst_threshold = detection_config['burst_threshold']
        error_threshold = detection_config['error_ratio_threshold']

        # Normalize burst score (0-1 scale)
        df = df.withColumn(
            "burst_score",
            when(col("request_count") >= burst_threshold, 1.0)
            .when(col("request_count") >= burst_threshold / 2,
                 (col("request_count") - burst_threshold / 2) / (burst_threshold / 2))
            .otherwise(0.0)
        )

        # Normalize error score (0-1 scale)
        df = df.withColumn(
            "error_score",
            when(col("error_ratio") >= error_threshold, 1.0)
            .when(col("error_ratio") > 0,
                 col("error_ratio") / error_threshold)
            .otherwise(0.0)
        )

        # User-Agent score is already 0-1 ratio
        df = df.withColumn("ua_score", col("ua_suspicious_ratio"))

        # Weighted bot score: burst (40%) + errors (30%) + UA (30%)
        df = df.withColumn(
            "bot_score",
            (col("burst_score") * 0.4 +
             col("error_score") * 0.3 +
             col("ua_score") * 0.3)
        )

        # Determine if IP is likely a bot
        df = df.withColumn(
            "is_bot",
            when(col("bot_score") >= detection_config['bot_score_threshold'], True)
            .otherwise(False)
        )

        # Generate detection reasons
        df = df.withColumn(
            "detection_reasons",
            concat_ws(", ",
                     when(col("burst_score") > 0.5, lit("high_request_rate")).otherwise(lit("")),
                     when(col("error_score") > 0.5, lit("high_error_ratio")).otherwise(lit("")),
                     when(col("ua_score") > 0.5, lit("suspicious_user_agent")).otherwise(lit("")))
        )

        # Add metadata
        df = df.withColumn("detection_timestamp", current_timestamp())

        return df

    def write_to_console(self, df):
        """Write streaming results to console for debugging"""
        query = (df
                 .writeStream
                 .outputMode("update")
                 .format("console")
                 .option("truncate", False)
                 .start())

        return query

    def write_to_elasticsearch(self, df, index_name):
        """Write streaming results to Elasticsearch"""
        es_config = self.config['elasticsearch']
        es_url = f"{es_config['host']}:{es_config['port']}"

        # Flatten window column for Elasticsearch
        df_flat = df.select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "*"
        ).drop("window")

        query = (df_flat
                 .writeStream
                 .outputMode("update")
                 .format("org.elasticsearch.spark.sql")
                 .option("es.nodes", es_config['host'])
                 .option("es.port", es_config['port'])
                 .option("es.resource", index_name)
                 .option("es.mapping.id", "client_ip")
                 .option("checkpointLocation", f"{self.config['checkpoint']['location']}/{index_name}")
                 .start())

        return query

    def run(self):
        """Main execution pipeline"""
        print("Starting WebBotShield Bot Detector...")
        print(f"Kafka: {self.config['kafka']['bootstrap_servers']}")
        print(f"Elasticsearch: {self.config['elasticsearch']['host']}:{self.config['elasticsearch']['port']}")

        # Read stream from Kafka
        raw_logs = self.read_kafka_stream()

        # Enrich with detection features
        enriched_logs = self.detect_suspicious_user_agents(raw_logs)
        enriched_logs = self.detect_error_patterns(enriched_logs)

        # Analyze traffic patterns
        traffic_analysis = self.analyze_traffic_patterns(enriched_logs)

        # Calculate bot scores
        bot_detections = self.calculate_bot_score(traffic_analysis)

        # Filter only bot detections
        bot_alerts = bot_detections.filter(col("is_bot") == True)

        # Write all detections to console (for debugging)
        console_query = self.write_to_console(bot_detections)

        print("Bot detection pipeline started successfully!")
        print("Monitoring for suspicious traffic patterns...")
        print("=" * 80)

        # Wait for termination
        console_query.awaitTermination()


if __name__ == "__main__":
    detector = BotDetector()
    detector.run()
