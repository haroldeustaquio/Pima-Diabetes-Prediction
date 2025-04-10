# Pime Diabetes Prediction


## Overview

This project implements a complete machine learning pipeline to predict diabetes risk using the Pima Indians dataset. It covers data preprocessing, model building, validation, optimization, and deployment via a FastAPI-based API.

**Content:**
- [Scripts](#scripts)
- [Notebooks](#notebooks)
- [Deployment](#deployment)

**Online Project**  
For more details and updates about this project, visit [this website](https://haroldeustaquio.com/projects/pima-diabetes-prediction).

---

## Scripts

**data_preprocessing.py**  
- Loads and transforms raw data (replacing zero values with NaN for key features).  
- Generates 8 preprocessing combinations (applying/removing outliers, balancing, scaling).  
- Saves processed CSV files with descriptive names.

**pipeline.py**  
- Configures preprocessing using simple and iterative imputations via a ColumnTransformer.  
- Integrates a pre-trained classifier into a unified pipeline.  
- Trains the model using processed data and saves the pipeline.

---

## Notebooks

- **01_data_preprocessing.ipynb:** Data import, cleaning, exploration, and normalization.
- **02_data_model_validation.ipynb:** Splitting data, model evaluation with metrics, and comparison.
- **03_model_optimization.ipynb:** Hyperparameter tuning and robust model validation.

Diagrams illustrate the ML pipeline architecture and best model performance.

---

## Deployment

- **API Overview:**  
  A FastAPI-based service for diabetes prediction.
  
- **Key Endpoints:**  
  - **GET `/`**: Health check.  
  - **POST `/predict`**: Returns a binary prediction.  
  - **POST `/predict_proba`**: Returns class probability scores.
  
- **Deployment Methods:**  
  Includes instructions for local Docker deployment and scalable deployment on AWS Elastic Beanstalk.

