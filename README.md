# Production-Ready Medical Image API: Pneumonia Detection 🩺

An end-to-end containerized REST API that serves a PyTorch deep learning model for classifying chest X-rays. 

## 🚀 The Objective
Deploying clinical machine learning models safely into production environments. This project bridges the gap between medical image processing and software engineering by serving a fine-tuned ResNet18 model through a high-performance FastAPI backend, all fully containerized via Docker for seamless deployment.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision (Transfer Learning with ResNet18)
* **Backend:** FastAPI, Uvicorn, Python-Multipart
* **DevOps/Deployment:** Docker
* **Data Processing:** Pillow, Scikit-learn

## 📊 Model Performance
* Achieved **87.5% Validation Accuracy** within 5 epochs using a frozen feature extractor and customized dense layers for binary classification (Normal vs. Pneumonia).

## 💻 How to Run Locally (Docker)
You do not need Python or PyTorch installed to run this API. Simply use Docker:

1. Clone the repository:
   ```bash
   git clone https://github.com/AhadAshfaq/production-medical-api.git
   cd production-medical-api

2. Build the Docker container:
   docker build -t medical-api .

3. Run the container:
   docker run -p 8000:8000 medical-api

4. Open http://localhost:8000/docs in your browser to interact with the API via the automated Swagger UI.