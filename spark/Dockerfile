# Custom Spark image with WebBotShield dependencies
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    openjdk-11-jdk-headless \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install Spark
RUN mkdir -p /opt/spark && \
    cd /opt && \
    wget -q https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.0-bin-hadoop3.tgz && \
    rm spark-3.5.0-bin-hadoop3.tgz && \
    mv spark-3.5.0-bin-hadoop3/* /opt/spark/ && \
    rm -rf spark-3.5.0-bin-hadoop3

# Install Python dependencies for ML
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    numpy==1.26.4 \
    pyyaml==6.0.1 \
    python-dateutil==2.8.2 \
    kafka-python==2.0.2 \
    elasticsearch==8.11.0 \
    scikit-learn==1.7.2 \
    joblib==1.3.2 \
    pyarrow==15.0.0 \
    shap==0.49.1 \
    onnx==1.15.0 \
    onnxruntime==1.17.0 \
    skl2onnx==1.16.0

# Create spark user and directories
RUN useradd -m -u 1000 spark || true
RUN mkdir -p /home/spark && chown -R spark:spark /home/spark
RUN mkdir -p /opt/spark-apps/models && chown -R spark:spark /opt/spark-apps
RUN chown -R spark:spark /opt/spark

# Switch to spark user
USER spark

# Set working directory
WORKDIR /opt/spark-apps
