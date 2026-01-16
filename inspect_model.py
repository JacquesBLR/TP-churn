import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import tempfile
import os
import shutil

# Créer des données de test identiques à celles du test
test_data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45],
    'Account_Manager': [0, 1, 0, 1, 0],
    'Years': [1, 2, 3, 4, 5],
    'Num_Sites': [5, 8, 10, 12, 15],
    'Churn': [0, 0, 1, 1, 0]
})

X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
y = test_data['Churn']

# Entraîner le modèle
model = LogisticRegression()
model.fit(X, y)

# Sauvegarder dans un fichier temporaire
test_dir = tempfile.mkdtemp()
model_path = os.path.join(test_dir, 'test_model.pkl')
joblib.dump(model, model_path)

print(f"📁 Chemin du modèle: {model_path}")
print(f"📊 Taille du fichier: {os.path.getsize(model_path)} octets\n")

# Charger et inspecter le contenu
loaded_model = joblib.load(model_path)

print("=" * 50)
print("📦 CONTENU DU MODÈLE .pkl")
print("=" * 50)
print(f"Type du modèle: {type(loaded_model).__name__}")
print(f"Classes possibles: {loaded_model.classes_}")
print(f"Nombre de features: {loaded_model.n_features_in_}")
print(f"Noms des features: {list(loaded_model.feature_names_in_)}")
print(f"\nCoefficients (poids): {loaded_model.coef_[0]}")
print(f"Intercept (biais): {loaded_model.intercept_[0]:.4f}")
print(f"\nParamètres du modèle:")
for key, value in loaded_model.get_params().items():
    print(f"  - {key}: {value}")

# Test de prédiction
test_input = pd.DataFrame([[30, 1, 2, 8]], columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
prediction = loaded_model.predict(test_input)
proba = loaded_model.predict_proba(test_input)

print(f"\n🔮 Test de prédiction avec {test_input.values[0]}:")
print(f"  Prédiction: {prediction[0]} ({'Churn' if prediction[0] == 1 else 'Pas de churn'})")
print(f"  Probabilités: [Pas de churn: {proba[0][0]:.2%}, Churn: {proba[0][1]:.2%}]")

# Nettoyer
shutil.rmtree(test_dir)
print(f"\n✅ Répertoire temporaire supprimé")
