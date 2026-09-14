FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libexpat1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860
CMD ["streamlit", "run", "app/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
