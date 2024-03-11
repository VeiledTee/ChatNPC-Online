FROM python:3.10.11

# Install build tools including gcc
RUN apt-get update && apt-get install -y build-essential

# By default, listen on port 5000
EXPOSE 5000/tcp

# Set the working directory in the container
WORKDIR /app

# Copy all files from the local directory to the working directory in the container
COPY . /app

# Install any dependencies
RUN pip install -r requirements.txt

# Specify the command to run on container start
CMD [ "python", "./app.py" ]