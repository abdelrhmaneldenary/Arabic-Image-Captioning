# 1. Use Python 3.10 Slim
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /code

# 3. Copy requirements (Basic libs)
COPY ./requirements.txt /code/requirements.txt

# 4. Install Basic Dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Install PyTorch (CPU Version) separately
# We run this as a separate command to ensure the URL flag works perfectly
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 6. Copy App and Model
COPY ./app /code/app
COPY ./model /code/model

# 7. Run the Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]