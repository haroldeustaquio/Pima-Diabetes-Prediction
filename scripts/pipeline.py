import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

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
rf_model = RandomForestClassifier(max_depth=15, min_samples_leaf=4, n_estimators=50, random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

df = pd.read_csv('../data/processed/data_remove_yes_balance_yes_scale_no.csv')

X = df.drop(columns='outcome')
y = df['outcome']

pipeline.fit(X, y)

# Save the pipeline
joblib.dump(pipeline, '../models/pipeline.pkl')