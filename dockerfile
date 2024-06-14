FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/static && chmod -R 777 /app/static

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]