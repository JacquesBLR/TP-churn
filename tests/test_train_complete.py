import unittest
import pandas as pd
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
import tempfile
import shutil
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTrainScript(unittest.TestCase):
    """Tests unitaires pour le script train.py"""
    
    def setUp(self):
        """Préparer l'environnement de test"""
        # Créer un répertoire temporaire
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, 'data')
        os.makedirs(self.test_data_dir)
        
        # Créer un fichier CSV de test
        self.test_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'Account_Manager': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'Years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Num_Sites': [5, 8, 10, 12, 15, 18, 20, 22, 25, 28],
            'Churn': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        })
        
        self.test_csv_path = os.path.join(self.test_data_dir, 'train_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        self.test_model_path = os.path.join(self.test_data_dir, 'churn_model_clean.pkl')
    
    def tearDown(self):
        """Nettoyer après chaque test"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_csv_file_exists(self):
        """Tester que le fichier CSV d'entraînement existe"""
        self.assertTrue(os.path.exists('data/train_data.csv'), 
                       "Le fichier data/train_data.csv n'existe pas")
    
    def test_csv_has_correct_columns(self):
        """Tester que le CSV contient les bonnes colonnes"""
        dataset = pd.read_csv('data/train_data.csv')
        
        required_columns = ['Age', 'Account_Manager', 'Years', 'Num_Sites', 'Churn']
        for col in required_columns:
            self.assertIn(col, dataset.columns, f"La colonne {col} est manquante")
    
    def test_csv_not_empty(self):
        """Tester que le CSV contient des données"""
        dataset = pd.read_csv('data/train_data.csv')
        self.assertGreater(len(dataset), 0, "Le fichier CSV est vide")
    
    def test_features_extraction(self):
        """Tester l'extraction des features"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        
        self.assertEqual(X.shape[1], 4, "Nombre de features incorrect")
        self.assertEqual(len(X), len(dataset), "Nombre de lignes incorrect")
        self.assertEqual(list(X.columns), ['Age', 'Account_Manager', 'Years', 'Num_Sites'])
    
    def test_target_extraction(self):
        """Tester l'extraction de la variable cible"""
        dataset = pd.read_csv(self.test_csv_path)
        y = dataset['Churn']
        
        self.assertEqual(len(y), len(dataset), "Nombre de valeurs target incorrect")
        self.assertTrue(all(y.isin([0, 1])), "Les valeurs de Churn doivent être 0 ou 1")
    
    def test_model_creation(self):
        """Tester la création du modèle LogisticRegression"""
        model = LogisticRegression()
        
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LogisticRegression)
    
    def test_model_training(self):
        """Tester l'entraînement du modèle"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Vérifier que le modèle est entraîné
        self.assertTrue(hasattr(model, 'coef_'), "Le modèle n'a pas de coefficients")
        self.assertTrue(hasattr(model, 'intercept_'), "Le modèle n'a pas d'intercept")
        self.assertEqual(model.n_features_in_, 4, "Nombre de features incorrect")
    
    def test_model_save(self):
        """Tester la sauvegarde du modèle"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Sauvegarder le modèle
        joblib.dump(model, self.test_model_path)
        
        self.assertTrue(os.path.exists(self.test_model_path), 
                       "Le fichier du modèle n'a pas été créé")
        self.assertGreater(os.path.getsize(self.test_model_path), 0, 
                          "Le fichier du modèle est vide")
    
    def test_saved_model_can_be_loaded(self):
        """Tester que le modèle sauvegardé peut être rechargé"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, self.test_model_path)
        
        # Charger le modèle
        loaded_model = joblib.load(self.test_model_path)
        
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, LogisticRegression)
    
    def test_loaded_model_can_predict(self):
        """Tester que le modèle chargé peut faire des prédictions"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, self.test_model_path)
        
        # Charger et utiliser le modèle
        loaded_model = joblib.load(self.test_model_path)
        test_input = pd.DataFrame([[30, 1, 2, 8]], 
                                  columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        prediction = loaded_model.predict(test_input)
        
        self.assertIsNotNone(prediction)
        self.assertIn(prediction[0], [0, 1])
    
    def test_model_predictions_are_consistent(self):
        """Tester que les prédictions sont cohérentes avant et après sauvegarde"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        test_input = pd.DataFrame([[30, 1, 2, 8]], 
                                  columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        prediction_before = model.predict(test_input)[0]
        
        # Sauvegarder et recharger
        joblib.dump(model, self.test_model_path)
        loaded_model = joblib.load(self.test_model_path)
        prediction_after = loaded_model.predict(test_input)[0]
        
        self.assertEqual(prediction_before, prediction_after, 
                        "Les prédictions diffèrent après sauvegarde/chargement")
    
    def test_no_missing_values_in_features(self):
        """Tester qu'il n'y a pas de valeurs manquantes dans les features"""
        dataset = pd.read_csv('data/train_data.csv')
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        
        self.assertEqual(X.isnull().sum().sum(), 0, 
                        "Il y a des valeurs manquantes dans les features")
    
    def test_no_missing_values_in_target(self):
        """Tester qu'il n'y a pas de valeurs manquantes dans la target"""
        dataset = pd.read_csv('data/train_data.csv')
        y = dataset['Churn']
        
        self.assertEqual(y.isnull().sum(), 0, 
                        "Il y a des valeurs manquantes dans la target")
    
    def test_data_types_are_numeric(self):
        """Tester que toutes les colonnes sont numériques"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        
        for col in X.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[col]), 
                          f"La colonne {col} n'est pas numérique")
    
    def test_churn_values_are_binary(self):
        """Tester que les valeurs de Churn sont binaires (0 ou 1)"""
        dataset = pd.read_csv('data/train_data.csv')
        y = dataset['Churn']
        
        unique_values = set(y.unique())
        self.assertTrue(unique_values.issubset({0, 1}), 
                       f"Churn doit contenir uniquement 0 ou 1, trouvé: {unique_values}")
    
    def test_model_has_expected_attributes_after_training(self):
        """Tester que le modèle a les attributs attendus après entraînement"""
        dataset = pd.read_csv(self.test_csv_path)
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Vérifier les attributs essentiels
        self.assertTrue(hasattr(model, 'coef_'))
        self.assertTrue(hasattr(model, 'intercept_'))
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertTrue(hasattr(model, 'n_features_in_'))
        self.assertTrue(hasattr(model, 'feature_names_in_'))
        
        # Vérifier les valeurs
        self.assertEqual(model.coef_.shape[1], 4)
        self.assertEqual(len(model.classes_), 2)


if __name__ == '__main__':
    unittest.main()
