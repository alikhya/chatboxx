# Use the official lightweight Python image.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory in container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Run the Flask app (assumes app is named 'app' inside chatbot.py)
CMD ["gunicorn", "-b", ":8080", "chatbot:app"]
