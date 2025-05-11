#!/usr/bin/env python3
"""
Integration tests for model_evaluator.py

These tests perform actual API calls to Ollama and LMStudio.
They require running instances of these services to work properly.

Run with:
python test_integration.py
"""

import unittest
import os
import json
import tempfile
import time
import requests
import shutil

# Import the module to test
from model_evaluator import ModelEvaluator, TASK_SUMMARIZATION, TASK_TRANSLATION

# Test content for summarization and translation
TEST_CONTENT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and 
takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that 
are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI 
researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, 
automated decision-making, and competing at the highest level in strategic game systems. As machines become increasingly capable, 
tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect.
"""

# Reference summary for testing
TEST_REFERENCE_SUMMARY = """
Artificial intelligence (AI) refers to machine-demonstrated intelligence, distinct from human or animal intelligence. 
AI research focuses on intelligent agents that perceive environments and act to achieve goals. While AI was once defined 
by human-like cognitive abilities, researchers now frame it in terms of rationality. Applications include search engines, 
recommendation systems, speech recognition, self-driving cars, and strategic games. As AI capabilities advance, previously 
"intelligent" tasks are often no longer considered AI, a phenomenon called the AI effect.
"""

# Reference translation for testing
TEST_REFERENCE_TRANSLATION = """
La inteligencia artificial (IA) es la inteligencia demostrada por máquinas, a diferencia de la inteligencia mostrada por 
animales, incluidos los humanos. La investigación en IA se ha definido como el campo de estudio de agentes inteligentes, 
que se refiere a cualquier sistema que percibe su entorno y toma acciones que maximizan sus posibilidades de alcanzar sus objetivos.

El término "inteligencia artificial" se había utilizado anteriormente para describir máquinas que imitan y muestran habilidades 
cognitivas "humanas" asociadas con la mente humana, como "aprendizaje" y "resolución de problemas". Esta definición ha sido 
rechazada por importantes investigadores de IA que ahora describen la IA en términos de racionalidad y actuación racional, 
lo que no limita cómo se puede articular la inteligencia.

Las aplicaciones de IA incluyen motores de búsqueda web avanzados, sistemas de recomendación, comprensión del habla humana, 
automóviles autónomos, toma de decisiones automatizada y competencia al más alto nivel en sistemas de juegos estratégicos. 
A medida que las máquinas se vuelven cada vez más capaces, las tareas consideradas que requieren "inteligencia" a menudo 
se eliminan de la definición de IA, un fenómeno conocido como el efecto IA.
"""

class TestOllamaIntegration(unittest.TestCase):
    """Integration tests for Ollama platform"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise Exception("Ollama API not responding")
            
            # Get available models
            models = [model["name"] for model in response.json()["models"]]
            
            # Check if required models are available
            required_models = ["llama3.2:latest", "granite3.3:latest", "gemma3:latest"]
            available_required = [model for model in required_models if model in models]
            
            if len(available_required) < 2:
                raise Exception(f"Not enough required models available. Found: {available_required}")
            
            cls.models = available_required
            cls.evaluator_model = cls.models[0]  # Use first available model as evaluator
            cls.test_models = cls.models[1:]  # Use remaining models for testing
            
        except Exception as e:
            raise unittest.SkipTest(f"Ollama not available: {e}")
        
        # Create a temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directory and its contents
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_summarization(self):
        """Test summarization with Ollama"""
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=TEST_REFERENCE_SUMMARY,
            project_dir=self.test_dir,
            task_type=TASK_SUMMARIZATION
        )
        
        # Run evaluation with a subset of models (max 2) to keep test duration reasonable
        test_models = self.test_models[:min(2, len(self.test_models))]
        results = evaluator.run_evaluation(
            TEST_CONTENT,
            test_models,
            self.evaluator_model
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(test_models))
        for model in test_models:
            self.assertIn(model, results)
            self.assertIn("output", results[model])
            self.assertIn("metrics", results[model])
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)

    def test_translation(self):
        """Test translation with Ollama"""
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=TEST_REFERENCE_TRANSLATION,
            project_dir=self.test_dir,
            task_type=TASK_TRANSLATION,
            source_language="English",
            target_language="Spanish"
        )
        
        # Run evaluation with a subset of models (max 2) to keep test duration reasonable
        test_models = self.test_models[:min(2, len(self.test_models))]
        results = evaluator.run_evaluation(
            TEST_CONTENT,
            test_models,
            self.evaluator_model
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(test_models))
        for model in test_models:
            self.assertIn(model, results)
            self.assertIn("output", results[model])
            self.assertIn("metrics", results[model])
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)

class TestLMStudioIntegration(unittest.TestCase):
    """Integration tests for LMStudio platform"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests"""
        # Check if LMStudio is running
        try:
            response = requests.get("http://127.0.0.1:1234/v1/models")
            if response.status_code != 200:
                raise Exception("LMStudio API not responding")
            
            # Get available models
            models_data = response.json()
            if "data" not in models_data or not models_data["data"]:
                raise Exception("No models available in LMStudio")
            
            models = [model["id"] for model in models_data["data"]]
            
            # We need at least 2 models for testing
            if len(models) < 2:
                raise Exception(f"Not enough models available. Found: {models}")
            
            cls.models = models
            cls.evaluator_model = cls.models[0]  # Use first available model as evaluator
            cls.test_models = cls.models[1:]  # Use remaining models for testing
            
        except Exception as e:
            raise unittest.SkipTest(f"LMStudio not available: {e}")
        
        # Create a temporary directory for test outputs
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test directory and its contents
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_summarization(self):
        """Test summarization with LMStudio"""
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="lmstudio",
            reference_text=TEST_REFERENCE_SUMMARY,
            project_dir=self.test_dir,
            task_type=TASK_SUMMARIZATION
        )
        
        # Run evaluation with a subset of models (max 2) to keep test duration reasonable
        test_models = self.test_models[:min(2, len(self.test_models))]
        results = evaluator.run_evaluation(
            TEST_CONTENT,
            test_models,
            self.evaluator_model
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(test_models))
        for model in test_models:
            self.assertIn(model, results)
            self.assertIn("output", results[model])
            self.assertIn("metrics", results[model])
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)

    def test_translation(self):
        """Test translation with LMStudio"""
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="lmstudio",
            reference_text=TEST_REFERENCE_TRANSLATION,
            project_dir=self.test_dir,
            task_type=TASK_TRANSLATION,
            source_language="English",
            target_language="Spanish"
        )
        
        # Run evaluation with a subset of models (max 2) to keep test duration reasonable
        test_models = self.test_models[:min(2, len(self.test_models))]
        results = evaluator.run_evaluation(
            TEST_CONTENT,
            test_models,
            self.evaluator_model
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(test_models))
        for model in test_models:
            self.assertIn(model, results)
            self.assertIn("output", results[model])
            self.assertIn("metrics", results[model])
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)

if __name__ == '__main__':
    unittest.main()
