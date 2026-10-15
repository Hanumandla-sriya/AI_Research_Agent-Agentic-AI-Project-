# Use an official lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to optimize Python running in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# CRITICAL: Tell Gradio to bind to all network interfaces so we can access it from our browser
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY research_agent.py app.py ./

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["sh", "-c", "export GRADIO_SERVER_PORT=${PORT:-7860}; exec python app.py"]