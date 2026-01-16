import unittest
import pandas as pd
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
import tempfile
import shutil

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTrainModel(unittest.TestCase):
    """Tests unitaires pour l'entraînement du modèle de churn"""
    
    def setUp(self):
        """Préparer les données de test avant chaque test"""
        # Créer un répertoire temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        
        # Créer un fichier CSV de test
        self.test_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Account_Manager': [0, 1, 0, 1, 0],
            'Years': [1, 2, 3, 4, 5],
            'Num_Sites': [5, 8, 10, 12, 15],
            'Churn': [0, 0, 1, 1, 0]
        })
        
        self.test_csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        self.test_model_path = os.path.join(self.test_dir, 'test_model.pkl')
    
    def tearDown(self):
        """Nettoyer après chaque test"""
        # Supprimer le répertoire temporaire
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_load_data(self):
        """Tester le chargement des données"""
        dataset = pd.read_csv(self.test_csv_path)
        
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 5)
        self.assertIn('Age', dataset.columns)
        self.assertIn('Churn', dataset.columns)
    
    def test_feature_columns_exist(self):
        """Tester que les colonnes de features existent"""
        dataset = pd.read_csv(self.test_csv_path)
        
        required_features = ['Age', 'Account_Manager', 'Years', 'Num_Sites']
        for feature in required_features:
            self.assertIn(feature, dataset.columns)
    
    def test_target_column_exists(self):
        """Tester que la colonne target existe"""
        dataset = pd.read_csv(self.test_csv_path)
        
        self.assertIn('Churn', dataset.columns)
    
    def test_prepare_features_and_target(self):
        """Tester la préparation des features et de la target"""
        dataset = pd.read_csv(self.test_csv_path)
        
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(X.shape[1], 4)
        self.assertEqual(len(y), 5)
    
    def test_model_training(self):
        """Tester l'entraînement du modèle"""
        dataset = pd.read_csv(self.test_csv_path)
        
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
    
    def test_model_prediction(self):
        """Tester que le modèle peut faire des prédictions"""
        dataset = pd.read_csv(self.test_csv_path)
        
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Faire une prédiction
        test_input = pd.DataFrame([[30, 1, 2, 8]], columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        prediction = model.predict(test_input)
        
        self.assertIsNotNone(prediction)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], [0, 1])
    
    def test_model_save_and_load(self):
        """Tester la sauvegarde et le chargement du modèle"""
        dataset = pd.read_csv(self.test_csv_path)
        
        X = dataset[['Age', 'Account_Manager', 'Years', 'Num_Sites']]
        y = dataset['Churn']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Sauvegarder le modèle
        joblib.dump(model, self.test_model_path)
        print(self.test_model_path) 
        self.assertTrue(os.path.exists(self.test_model_path))
        
        # Charger le modèle
        loaded_model = joblib.load(self.test_model_path)
        self.assertIsNotNone(loaded_model)
        
        # Vérifier que le modèle chargé peut faire des prédictions
        test_input = pd.DataFrame([[30, 1, 2, 8]], columns=['Age', 'Account_Manager', 'Years', 'Num_Sites'])
        prediction = loaded_model.predict(test_input)
        self.assertIsNotNone(prediction)
    
    def test_data_types(self):
        """Tester les types de données des colonnes"""
        dataset = pd.read_csv(self.test_csv_path)
        
        # Vérifier que les colonnes numériques sont bien numériques
        self.assertTrue(pd.api.types.is_numeric_dtype(dataset['Age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(dataset['Years']))
        self.assertTrue(pd.api.types.is_numeric_dtype(dataset['Churn']))


if __name__ == '__main__':
    unittest.main()
