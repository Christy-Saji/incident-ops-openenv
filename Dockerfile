FROM python:3.11-slim

WORKDIR /app
COPY . .

# Server-side deps only (no torch/unsloth — training runs on Colab / AMD cloud).
# The base [project] dependencies in pyproject.toml are exactly the server set:
# fastapi, openai, pydantic, pyyaml, uvicorn.
RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
