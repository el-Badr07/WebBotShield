# Custom Spark image with WebBotShield dependencies
FROM apache/spark:3.5.0-python3

# Switch to root to install packages
USER root

# Install Python package manager and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    pandas==1.3.5 \
    numpy==1.21.6 \
    pyyaml==6.0.1 \
    python-dateutil==2.8.2 \
    kafka-python==2.0.2 \
    elasticsearch==8.11.0

# Create home directory for spark user if it doesn't exist
RUN mkdir -p /home/spark && chown -R spark:spark /home/spark

# Switch back to spark user
USER spark

# Set working directory
WORKDIR /opt/spark-apps
