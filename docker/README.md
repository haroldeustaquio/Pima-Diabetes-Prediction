# Pima Diabetes Prediction API 

## Overview

 FastAPI-based machine learning API for predicting diabetes risk using the Pima Indians dataset. The API is containerized with Docker and deployed on AWS Elastic Beanstalk for scalability.

**Content**
+ [Architecture](#architecture)
+ [API Endpoints](#api-endpoints)
+ [Local Deployment with Docker](#local-deployment-with-docker)
+ [AWS Elastic Beanstalk Deployment](#aws-elastic-beanstalk-deployment)

---

## Architecture

<div align="center">

<figure>
    <img src="https://github.com/user-attachments/assets/3e88c0c9-806a-4daf-a9fa-fde3e683ebcc" alt="Captioned Image">
    <figcaption>Architecture of the API, deploy in Docker and AWS</figcaption>
</figure>

</div>

---

## API Endpoints

Built with FastAPI, the API exposes:

**GET /**
- Health check: Returns {"message": "hola harold"}.

**POST /predict**
- Accepts a JSON payload with features (list of 8 numerical values).
Returns {"prediction": 0/1}.

**POST /predict_proba**
- Returns probability scores for both classes, e.g., {"prediction_proba": [0.2, 0.8]}

---

## Local Deployment with Docker

1. **Build the Docker Image**

```bash
docker build -t diabetes_prediction -f Dockerfile .
```

2. **Run the container**

```bash
docker run -p 8000:8000 diabetes_prediction
```

3. **Test the API**

Access ``http://localhost:8000/docs`` for interactive Swagger documentation.

---

## AWS Elastic Beanstalk Deployment

1. **Install and configure EB CLI**

```bash
pip install awsebcli --upgrade --user  
aws configure  # Set AWS credentials  
```

2. **Initialize Elastic Beanstalk**

```bash
eb init -p docker diabetes-prediction-api  
```

3. **Create and deploy the environment**

```bash
eb create diabetes-env  
```

4. **Access the deployed API**

Use the generated URL

---
