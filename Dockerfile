# Stage 1: Build environment
FROM python:3.10.11 AS builder

# Install build tools including gcc
RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt to avoid cache invalidation
COPY requirements.txt /app

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Stage 2: Runtime environment
FROM python:3.10.11-slim

# Expose port 8080
EXPOSE 8080/tcp

# Set the working directory in the container
WORKDIR /app

# Copy built files from the previous stage
COPY --from=builder /app /app

# Specify the command to run on container start
CMD ["python", "./app.py"]