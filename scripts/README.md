# Scripts

## Overview

This set of files is part of the processing and modeling pipeline for predicting diabetes using the Pima dataset. Below is the documentation for two key scripts that enable data transformation and the construction of a machine learning pipeline.

**Content**
- [data_preprocessing.py](#data_preprocessingpy)
- [pipeline.py](#pipelinepy)

---

## data_preprocessing.py

### Description
This script is responsible for preparing the raw data from the file `original.csv` for modeling. Its main functionalities include:

1. **Initial Data Loading and Transformation**  
    - Loads the original dataset and replaces zero values with `np.nan` in critical columns such as *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, and *BMI*.  
    - Converts all column names to lowercase for uniformity.

2. **Generation of Preprocessing Combinations**  
    - Utilizes the Cartesian product of options to create 8 different combinations for applying or omitting outlier removal, data balancing, and variable scaling.

3. **Application of Preprocessing Techniques**  
    - **Imputation:** Uses imputation recommendations based on the `functions_preprocessing` module.  
    - **Outlier Removal:** Applies this step conditionally based on the selected option.  
    - **Data Balancing:** Implements oversampling techniques to adjust class distribution, sometimes using a specific method such as *RandomOverSampler* with defined count targets.  
    - **Scaling:** Scales numerical variables as per the provided recommendation.

4. **Generation and Storage of Processed Files**  
    - For each preprocessing combination, the script saves a CSV file with a descriptive name indicating which techniques were applied (for example, `data_remove_yes_balance_no_scale_yes.csv`).


---

## pipeline.py

### Description
This file builds and integrates a machine learning pipeline that combines data preprocessing and the classifier model. Its main functionalities are:

1. **Preprocessing Configuration**  
    - Two groups of columns are defined for imputation:  
      - **Simple Imputation:** For columns *glucose*, *bloodpressure*, and *insulin*, using `SimpleImputer` with the median strategy.  
      - **Iterative Imputation:** For columns *skinthickness* and *bmi*, using `IterativeImputer` with a fixed random state.  
    - A `ColumnTransformer` is used to apply these transformations selectively, while other columns remain unchanged.

2. **Loading the Optimized Model**  
    - Loads the trained model (stored as `best_model.pkl`), which is typically an ensemble classifier (e.g., RandomForestClassifier).

3. **Pipeline Construction**  
    - Creates a pipeline that integrates preprocessing and the classifier, encapsulating the entire workflow in a single process.

4. **Training and Storage**  
    - Reads the preprocessed dataset (`processed.csv`), separates the features (X) and the target variable (y), and trains the pipeline.  
    - The trained pipeline is saved in specific locations for later use in production and Docker-based environments, ensuring a modular and straightforward integration.
