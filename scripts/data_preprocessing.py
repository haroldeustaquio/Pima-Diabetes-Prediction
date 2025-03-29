from importlib import reload
import pandas as pd
import numpy as np
import functions_preprocessing as f
from itertools import product

reload(f)

df_original = pd.read_csv('../data/original/data.csv')

df_original[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_original[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df_original.columns = df_original.columns.str.lower()

num_features = df_original.drop(columns='outcome').select_dtypes(include=['number']).columns

options = list(product([True, False], repeat=3))

for remove_outliers, balance_data, scale_data in options:
    df = df_original.copy()
    
    # 1. Imputar valores nulos
    imputer_recommendations = f.imputer_recommender(df, num_features, soft=False, threshold_null=0.5)
    df = f.apply_imputer_strategy(df, imputer_recommendations)
    
    # 2. Eliminar outliers (si se selecciona)
    if remove_outliers:
        f.outlier_handling_detect_all(df, num_features, remove=True)
    
    # 3. Balancear los datos (si se selecciona)
    if balance_data:
        X = df.drop(columns='outcome')
        y = df['outcome']
        balancing_recommendation = f.data_balancing_oversampling_recommender(X, y, soft=False)
        df = f.apply_balancing_oversampling(X, y, balancing_recommendation)
    
        X = df.drop(columns='outcome')
        y = df['outcome']
        df = f.apply_balancing_oversampling(X, y, recommendation="RandomOverSampler", target_counts={0: 500, 1: 500})
    
    # 4. Escalar las características (si se selecciona)
    if scale_data:
        suggest_transform = f.feature_scaling_recommender(df, num_features, soft=False)
        df = f.apply_scaling_transform(df, suggest_transform)
    
    # Construir un nombre de archivo descriptivo
    filename = (
        f"../data/processed/data_remove_{'yes' if remove_outliers else 'no'}_"
        f"balance_{'yes' if balance_data else 'no'}_"
        f"scale_{'yes' if scale_data else 'no'}.csv"
    )
    
    df.to_csv(filename, index=False)
    print(f"Archivo guardado: {filename}")