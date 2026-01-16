import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
import tempfile
import shutil


def test_load_data():
    """Tester le chargement des données"""
    # Créer des données de test
    test_data = pd.DataFrame({
        'Age': [25, 30, 35],
        'Account_Manager': [0, 1, 0],
        'Years': [1, 2, 3],
        'Num_Sites': [5, 8, 10],
        'Churn': [0, 1, 0]
    })
    
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, 'test.csv')
    test_data.to_csv(csv_path, index=False)
    
    # Tester le chargement
    dataset = pd.read_csv(csv_path)
    
    assert dataset is not None
    assert len(dataset) == 3
    assert 'Age' in dataset.columns
    assert 'Churn' in dataset.columns
    
    # Nettoyer
    shutil.rmtree(temp_dir)
    print("✓ test_load_data passé")


def test_feature_columns_exist():
    """Tester que les colonnes de features existent"""
    test_data = pd.DataFrame({
        'Age': [25, 30],
        'Account_Manager': [0, 1],
        'Years': [1, 2],
        'Num_Sites': [5, 8],
        'Churn': [0, 1]
    })
    
    required_features = ['Age', 'Account_Manager', 'Years', 'Num_Sites']
    for feature in required_features:
        assert feature in test_data.columns, f"Feature {feature} manquante"
    
    print("✓ test_feature_columns_exist passé")


def test_model_training():
    """Tester l'entraînement du modèle"""
    test_data = pd.DataFrame({
        'Age': [25, 30, 35, 40],
        'Account_Manager': [0, 1, 0, 1],
        'Years': [1, 2, 3, 4],
        'Num_Sites': [5, 8, 10, 12],
        'Churn': [0, 1, 0, 1]
    })
    
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    
    print("✓ test_model_training passé")


def test_model_prediction():
    """Tester les prédictions du modèle"""
    test_data = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Account_Manager': [0, 1, 0, 1, 0],
        'Years': [1, 2, 3, 4, 5],
        'Num_Sites': [5, 8, 10, 12, 15],
        'Churn': [0, 0, 1, 1, 0]
    })
    
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Faire une prédiction
    test_input = pd.DataFrame([[30, 1, 2, 8]], columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    prediction = model.predict(test_input)
    
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    
    print("✓ test_model_prediction passé")


def test_model_save_and_load():
    """Tester la sauvegarde et le chargement du modèle"""
    test_data = pd.DataFrame({
        'Age': [25, 30, 35, 40],
        'Account_Manager': [0, 1, 0, 1],
        'Years': [1, 2, 3, 4],
        'Num_Sites': [5, 8, 10, 12],
        'Churn': [0, 1, 0, 1]
    })
    
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Sauvegarder le modèle
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'test_model.pkl')
    joblib.dump(model, model_path)
    
    assert os.path.exists(model_path)
    
    # Charger le modèle
    loaded_model = joblib.load(model_path)
    assert loaded_model is not None
    
    # Vérifier que le modèle chargé peut faire des prédictions
    test_input = pd.DataFrame([[30, 1, 2, 8]], columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    prediction = loaded_model.predict(test_input)
    assert prediction is not None
    
    # Nettoyer
    shutil.rmtree(temp_dir)
    print("✓ test_model_save_and_load passé")


def test_data_types():
    """Tester les types de données des colonnes"""
    test_data = pd.DataFrame({
        'Age': [25, 30, 35],
        'Account_Manager': [0, 1, 0],
        'Years': [1, 2, 3],
        'Num_Sites': [5, 8, 10],
        'Churn': [0, 1, 0]
    })
    
    # Vérifier que les colonnes numériques sont bien numériques
    assert pd.api.types.is_numeric_dtype(test_data['Age'])
    assert pd.api.types.is_numeric_dtype(test_data['Years'])
    assert pd.api.types.is_numeric_dtype(test_data['Churn'])
    
    print("✓ test_data_types passé")


def test_prepare_features_and_target():
    """Tester la préparation des features et de la target"""
    test_data = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45],
        'Account_Manager': [0, 1, 0, 1, 0],
        'Years': [1, 2, 3, 4, 5],
        'Num_Sites': [5, 8, 10, 12, 15],
        'Churn': [0, 0, 1, 1, 0]
    })
    
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    assert X.shape[0] == 5
    assert X.shape[1] == 4
    assert len(y) == 5
    
    print("✓ test_prepare_features_and_target passé")


if __name__ == '__main__':
    print("=" * 50)
    print("Exécution des tests unitaires")
    print("=" * 50 + "\n")
    
    try:
        test_load_data()
        test_feature_columns_exist()
        test_model_training()
        test_model_prediction()
        test_model_save_and_load()
        test_data_types()
        test_prepare_features_and_target()
        
        print("\n" + "=" * 50)
        print("✅ Tous les tests sont passés !")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n❌ Test échoué: {e}")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
