FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#buat service accountnya dulu ygy
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/path/to/your/service-account-file.json"

EXPOSE 8000

CMD ["python", "app.py"]
