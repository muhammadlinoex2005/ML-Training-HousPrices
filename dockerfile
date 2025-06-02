# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (app.py, templates folder, .pkl files, etc.)
COPY . .

# Define the command to run your app using Gunicorn
# Hugging Face Spaces often injects a PORT environment variable, typically 7860 for Docker apps.
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "0"]