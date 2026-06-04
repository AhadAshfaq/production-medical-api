# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install PyTorch CPU version explicitly to keep the container size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Copy requirements and install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the FastAPI app logic and the model weights
COPY ./app ./app
COPY model_weights.pth .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]