# Custom Spark image with WebBotShield dependencies
FROM apache/spark:3.5.0-python3

# Switch to root to install packages
USER root

# Install Python package manager and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ML
RUN pip3 install --no-cache-dir \
    pandas==2.0.3 \
    numpy==1.24.3 \
    pyyaml==6.0.1 \
    python-dateutil==2.8.2 \
    kafka-python==2.0.2 \
    elasticsearch==8.11.0 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    pyarrow==14.0.1

# Create directories
RUN mkdir -p /home/spark && chown -R spark:spark /home/spark
RUN mkdir -p /opt/spark-apps/models && chown -R spark:spark /opt/spark-apps

# Switch back to spark user
USER spark

# Set working directory
WORKDIR /opt/spark-apps
