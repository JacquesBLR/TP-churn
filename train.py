import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

dataset = pd.read_csv('data/train_data.csv')

X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]

y = dataset['Churn']

model = LogisticRegression()

model.fit(X, y)

# Sauvegarde du modèle entraîné

joblib.dump(model, 'data/churn_model_clean.pkl')
print("Modèle entraîné et sauvegardé avec succès.")

