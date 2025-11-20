# Fluentd Dockerfile with Kafka plugin
FROM fluent/fluentd:v1.16-1

USER root

# Install build dependencies required for native gem extensions
RUN apk add --no-cache ruby-dev build-base

# Install required plugins
RUN gem install fluent-plugin-kafka --no-document

USER fluent
