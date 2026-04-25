FROM python:3.11-slim

WORKDIR /app
COPY . .

# Only install server-side deps (no torch/unsloth — those run in Colab)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
