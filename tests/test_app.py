import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from app import app
import json


class TestFlaskApp(unittest.TestCase):
    """Tests unitaires pour l'application Flask app.py"""
    
    def setUp(self):
        """Configurer le client de test Flask avant chaque test"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_home_route_status_code(self):
        """Tester que la route / retourne un code 200"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_home_route_returns_html(self):
        """Tester que la route / retourne du HTML"""
        response = self.client.get('/')
        self.assertEqual(response.content_type, 'text/html; charset=utf-8')
    
    def test_home_route_contains_html_content(self):
        """Tester que la page d'accueil contient du contenu HTML"""
        response = self.client.get('/')
        self.assertIn(b'<!DOCTYPE html>', response.data)
    
    def test_predict_route_exists(self):
        """Tester que la route /predict existe"""
        response = self.client.post('/predict', data={})
        # Devrait retourner une erreur (400 ou 500) mais pas 404
        self.assertNotEqual(response.status_code, 404)
    
    def test_predict_with_valid_data(self):
        """Tester une prédiction avec des données valides"""
        data = {
            'Age': 45,
            'Account_Manager': 1,
            'Years': 5,
            'Num_Sites': 10
        }
        response = self.client.post('/predict', data=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'application/json')
        
        json_data = json.loads(response.data)
        self.assertIn('churn_prediction', json_data)
        self.assertIn(json_data['churn_prediction'], [0, 1])
    
    def test_predict_with_missing_fields(self):
        """Tester une prédiction avec des champs manquants"""
        data = {
            'Age': 45,
            'Account_Manager': 1
            # Years et Num_Sites manquants
        }
        response = self.client.post('/predict', data=data)
        
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
    
    def test_predict_with_invalid_data_type(self):
        """Tester une prédiction avec des types de données invalides"""
        data = {
            'Age': 'invalid',  # Devrait être un nombre
            'Account_Manager': 1,
            'Years': 5,
            'Num_Sites': 10
        }
        response = self.client.post('/predict', data=data)
        
        json_data = json.loads(response.data)
        self.assertIn('error', json_data)
    
    def test_predict_returns_json(self):
        """Tester que /predict retourne du JSON"""
        data = {
            'Age': 30,
            'Account_Manager': 0,
            'Years': 2,
            'Num_Sites': 8
        }
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.content_type, 'application/json')
    
    def test_predict_with_boundary_values(self):
        """Tester des prédictions avec des valeurs limites"""
        # Valeurs minimales
        data = {
            'Age': 18,
            'Account_Manager': 0,
            'Years': 0,
            'Num_Sites': 1
        }
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        
        # Valeurs élevées
        data = {
            'Age': 100,
            'Account_Manager': 1,
            'Years': 50,
            'Num_Sites': 100
        }
        response = self.client.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
    
    def test_model_loaded(self):
        """Tester que le modèle est chargé au démarrage"""
        from app import model
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
    
    def test_predict_multiple_predictions(self):
        """Tester plusieurs prédictions consécutives"""
        test_cases = [
            {'Age': 25, 'Account_Manager': 0, 'Years': 1, 'Num_Sites': 5},
            {'Age': 50, 'Account_Manager': 1, 'Years': 10, 'Num_Sites': 15},
            {'Age': 35, 'Account_Manager': 1, 'Years': 3, 'Num_Sites': 8}
        ]
        
        for data in test_cases:
            response = self.client.post('/predict', data=data)
            self.assertEqual(response.status_code, 200)
            json_data = json.loads(response.data)
            self.assertIn('churn_prediction', json_data)


if __name__ == '__main__':
    unittest.main()
