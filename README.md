# Image Classification with XAI – Automated Malaria Detection
## Overview
This project focuses on **automated malaria detection** using image classification techniques powered by advanced Convolutional Neural Network (CNN) architectures. Multiple CNN model variants were trained and evaluated to enhance the accuracy and efficiency of malaria parasite identification from microscopic images.
To ensure model transparency and interpretability, **Explainable AI (XAI)** methods were integrated into the workflow. Specifically, **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-Agnostic Explanations)** were employed to visualize and explain model predictions. These XAI techniques highlight important image regions influencing the classification outcome, allowing medical professionals to trust and validate the AI’s decision-making process.


## Explainable AI (XAI)
Explainable AI techniques used:

###  SHAP
- Computes **feature importance** based on Shapley values  
- Highlights **critical regions** in the image influencing predictions  



### LIME
- Provides **local interpretability** for individual predictions  
- Creates **superpixel visualizations** to show what areas drive classification decisions  




## Model Performance

| Model      | Accuracy | Precision | Recall  | F1-Score |
|------------|---------|-----------|--------|----------|
| CNN        | 0.9526  | 0.9847    | 0.9202 | 0.9508   |
| ResNet     | 0.9675  | 0.9771    | 0.9572 | 0.9656   |
| MobileNet  | 0.9660  | 0.9834    | 0.9478 | 0.9653   |
| VGG        | 0.9310  | 0.9648    | 0.8941 | 0.9281   |

**Observations:**  
- **ResNet** achieved the highest accuracy and F1-score overall.  
- **MobileNet** also performed very well, with slightly lower recall than ResNet.



## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
- python (>= 3.12)

### Installation
```bash
# Clone the repository
git clone https://github.com/Amin-0513/Image-Classification-with-XAI.git

# Navigate to project directory
cd Image-Classification-with-XAI

# create python environment
python -m venv xai

# activate python environment
xai\Scripts\activate

# Install dependencies
pip install -r requirments.txt

## Start project
streamlit run app.py

```
## Dockerfile

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p temp_images

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Build Docker Image

```bash
docker build -t malaria-xai-app .
```

## Run Docker Container

```bash
•	docker run -d -p 8501:8501 --name malaria-xai malaria-xai-app
```

## Access the Application

Open the following URL in your browser:

http://localhost:8501

## View Container Logs

```bash
docker logs -f malaria-xai
```

## Stop the Container

```bash
docker stop malaria-xai
```

## Remove the Container

```bash
docker rm -f malaria-xai
```
## Docker Hub
```bash
docker push pda2/malaria-xai-app:latest
```
