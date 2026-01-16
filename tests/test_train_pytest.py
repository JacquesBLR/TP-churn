import pytest
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
import tempfile
import shutil


@pytest.fixture
def test_data():
    """Fixture pour créer un DataFrame de test"""
    return pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'Account_Manager': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Num_Sites': [5, 8, 10, 12, 15, 18, 20, 22, 25, 28],
        'Churn': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    })


@pytest.fixture
def temp_dir():
    """Fixture pour créer un répertoire temporaire"""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def test_csv_path(test_data, temp_dir):
    """Fixture pour créer un fichier CSV de test"""
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(data_dir)
    csv_path = os.path.join(data_dir, 'train_data.csv')
    test_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def trained_model(test_data):
    """Fixture pour créer un modèle entraîné"""
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def model_path(trained_model, temp_dir):
    """Fixture pour sauvegarder un modèle dans un fichier temporaire"""
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    model_file = os.path.join(data_dir, 'churn_model_clean.pkl')
    joblib.dump(trained_model, model_file)
    return model_file


# ===== Tests de données =====

def test_csv_file_exists():
    """Tester que le fichier CSV d'entraînement existe"""
    assert os.path.exists('data/train_data.csv'), "Le fichier data/train_data.csv n'existe pas"


def test_csv_has_correct_columns():
    """Tester que le CSV contient les bonnes colonnes"""
    dataset = pd.read_csv('data/train_data.csv')
    required_columns = ['Age', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']
    
    for col in required_columns:
        assert col in dataset.columns, f"La colonne {col} est manquante"


def test_csv_not_empty():
    """Tester que le CSV contient des données"""
    dataset = pd.read_csv('data/train_data.csv')
    assert len(dataset) > 0, "Le fichier CSV est vide"


def test_no_missing_values_in_features():
    """Tester qu'il n'y a pas de valeurs manquantes dans les features"""
    dataset = pd.read_csv('data/train_data.csv')
    X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    assert X.isnull().sum().sum() == 0, "Il y a des valeurs manquantes dans les features"


def test_no_missing_values_in_target():
    """Tester qu'il n'y a pas de valeurs manquantes dans la target"""
    dataset = pd.read_csv('data/train_data.csv')
    y = dataset['Churn']
    assert y.isnull().sum() == 0, "Il y a des valeurs manquantes dans la target"


def test_churn_values_are_binary():
    """Tester que les valeurs de Churn sont binaires (0 ou 1)"""
    dataset = pd.read_csv('data/train_data.csv')
    y = dataset['Churn']
    unique_values = set(y.unique())
    assert unique_values.issubset({0, 1}), f"Churn doit contenir uniquement 0 ou 1, trouvé: {unique_values}"


# ===== Tests d'extraction =====

def test_features_extraction(test_csv_path):
    """Tester l'extraction des features"""
    dataset = pd.read_csv(test_csv_path)
    X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    
    assert X.shape[1] == 4, "Nombre de features incorrect"
    assert len(X) == len(dataset), "Nombre de lignes incorrect"
    assert list(X.columns) == ['Age', 'Account_Manager', 'Years', 'Num_Sites']


def test_target_extraction(test_csv_path):
    """Tester l'extraction de la variable cible"""
    dataset = pd.read_csv(test_csv_path)
    y = dataset['Churn']
    
    assert len(y) == len(dataset), "Nombre de valeurs target incorrect"
    assert all(y.isin([0, 1])), "Les valeurs de Churn doivent être 0 ou 1"


def test_data_types_are_numeric(test_csv_path):
    """Tester que toutes les colonnes sont numériques"""
    dataset = pd.read_csv(test_csv_path)
    X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"La colonne {col} n'est pas numérique"


# ===== Tests du modèle =====

def test_model_creation():
    """Tester la création du modèle LogisticRegression"""
    model = LogisticRegression()
    
    assert model is not None
    assert isinstance(model, LogisticRegression)


def test_model_training(test_data):
    """Tester l'entraînement du modèle"""
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    assert hasattr(model, 'coef_'), "Le modèle n'a pas de coefficients"
    assert hasattr(model, 'intercept_'), "Le modèle n'a pas d'intercept"
    assert model.n_features_in_ == 4, "Nombre de features incorrect"


def test_model_has_expected_attributes_after_training(trained_model):
    """Tester que le modèle a les attributs attendus après entraînement"""
    assert hasattr(trained_model, 'coef_')
    assert hasattr(trained_model, 'intercept_')
    assert hasattr(trained_model, 'classes_')
    assert hasattr(trained_model, 'n_features_in_')
    assert hasattr(trained_model, 'feature_names_in_')
    
    assert trained_model.coef_.shape[1] == 4
    assert len(trained_model.classes_) == 2


# ===== Tests de sauvegarde =====

def test_model_save(test_data, temp_dir):
    """Tester la sauvegarde du modèle"""
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    model_file = os.path.join(temp_dir, 'test_model.pkl')
    joblib.dump(model, model_file)
    
    assert os.path.exists(model_file), "Le fichier du modèle n'a pas été créé"
    assert os.path.getsize(model_file) > 0, "Le fichier du modèle est vide"


def test_saved_model_can_be_loaded(model_path):
    """Tester que le modèle sauvegardé peut être rechargé"""
    loaded_model = joblib.load(model_path)
    
    assert loaded_model is not None
    assert isinstance(loaded_model, LogisticRegression)


def test_loaded_model_can_predict(model_path):
    """Tester que le modèle chargé peut faire des prédictions"""
    loaded_model = joblib.load(model_path)
    test_input = pd.DataFrame([[30, 1, 2, 8]], 
                              columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    prediction = loaded_model.predict(test_input)
    
    assert prediction is not None
    assert prediction[0] in [0, 1]


def test_model_predictions_are_consistent(model_path, test_data):
    """Tester que les prédictions sont cohérentes avant et après sauvegarde"""
    X = test_data[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
    y = test_data['Churn']
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    test_input = pd.DataFrame([[30, 1, 2, 8]], 
                              columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    prediction_before = model.predict(test_input)[0]
    
    # Charger le modèle sauvegardé
    loaded_model = joblib.load(model_path)
    prediction_after = loaded_model.predict(test_input)[0]
    
    assert prediction_before == prediction_after, \
        "Les prédictions diffèrent après sauvegarde/chargement"


# ===== Tests de prédiction =====

@pytest.mark.parametrize("age,account_manager,years,num_sites", [
    (25, 0, 1, 5),
    (50, 1, 10, 15),
    (35, 1, 3, 8),
    (60, 0, 8, 20),
])
def test_model_predictions_with_various_inputs(trained_model, age, account_manager, years, num_sites):
    """Tester les prédictions avec différentes valeurs d'entrée"""
    test_input = pd.DataFrame([[age, account_manager, years, num_sites]], 
                              columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    prediction = trained_model.predict(test_input)
    
    assert prediction is not None
    assert prediction[0] in [0, 1]


def test_model_predict_proba(trained_model):
    """Tester que le modèle peut retourner des probabilités"""
    test_input = pd.DataFrame([[30, 1, 2, 8]], 
                              columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    probas = trained_model.predict_proba(test_input)
    
    assert probas is not None
    assert probas.shape == (1, 2)
    assert abs(probas.sum() - 1.0) < 0.001  # Les probabilités doivent sommer à 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
