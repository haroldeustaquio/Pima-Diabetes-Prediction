import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import cloudpickle

# From notebooks/01_data_preprocessing.ipynb
simple_cols = ['glucose', 'bloodpressure', 'insulin']
iterative_cols = ['skinthickness', 'bmi']

simple_imputer = SimpleImputer(strategy='median')
iterative_imputer = IterativeImputer(random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('simple_imp', simple_imputer, simple_cols),
        ('iterative_imp', iterative_imputer, iterative_cols)
    ],
    remainder='passthrough'
)

# From notebooks/03_model_optimization.ipynb
rf_model = joblib.load('../models/best_model.pkl')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

df = pd.read_csv('../data/processed.csv')

X = df.drop(columns='outcome')
y = df['outcome']

pipeline.fit(X, y)


joblib.dump(pipeline, "../models/pipeline.joblib")
joblib.dump(pipeline, "../docker/app/pipeline.joblib")