#!/usr/bin/env python3
"""
Unit tests for model_evaluator.py
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import the module to test
from model_evaluator import ModelEvaluator, TASK_SUMMARIZATION, TASK_TRANSLATION

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_content = "This is a test content for summarization and translation."
        self.test_reference = "This is a reference summary or translation."
        self.test_output = "This is a generated output from a model."
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Sample models for testing
        self.ollama_models = ["llama3.2:latest", "granite3.3:latest"]
        self.lmstudio_models = ["llama3.2", "granite3.3"]
        self.evaluator_model = "gemma3:latest"
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directory and its contents
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    @patch('requests.post')
    @patch('requests.get')
    def test_ollama_summarization(self, mock_get, mock_post):
        """Test summarization with Ollama models"""
        # Mock the API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "granite3.3:latest"},
                {"name": "gemma3:latest"}
            ]
        }
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": self.test_output
        }
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=self.test_reference,
            project_dir=self.test_dir,
            task_type=TASK_SUMMARIZATION
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            self.test_content,
            ["llama3.2:latest", "granite3.3:latest"],
            "gemma3:latest"
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("llama3.2:latest", results)
        self.assertIn("granite3.3:latest", results)
        self.assertEqual(results["llama3.2:latest"]["output"], self.test_output)
        self.assertEqual(results["granite3.3:latest"]["output"], self.test_output)
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)
        
        # Verify API calls
        self.assertTrue(mock_get.called)
        self.assertTrue(mock_post.called)

    @patch('requests.post')
    @patch('requests.get')
    def test_ollama_translation(self, mock_get, mock_post):
        """Test translation with Ollama models"""
        # Mock the API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "granite3.3:latest"},
                {"name": "gemma3:latest"}
            ]
        }
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": self.test_output
        }
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=self.test_reference,
            project_dir=self.test_dir,
            task_type=TASK_TRANSLATION,
            source_language="English",
            target_language="Spanish"
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            self.test_content,
            ["llama3.2:latest", "granite3.3:latest"],
            "gemma3:latest"
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("llama3.2:latest", results)
        self.assertIn("granite3.3:latest", results)
        self.assertEqual(results["llama3.2:latest"]["output"], self.test_output)
        self.assertEqual(results["granite3.3:latest"]["output"], self.test_output)
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)
        
        # Verify API calls
        self.assertTrue(mock_get.called)
        self.assertTrue(mock_post.called)

    @patch('requests.post')
    @patch('requests.get')
    def test_lmstudio_summarization(self, mock_get, mock_post):
        """Test summarization with LMStudio models"""
        # Mock the API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "llama3.2"},
                {"id": "granite3.3"},
                {"id": "gemma3"}
            ]
        }
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"text": self.test_output}]
        }
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="lmstudio",
            reference_text=self.test_reference,
            project_dir=self.test_dir,
            task_type=TASK_SUMMARIZATION
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            self.test_content,
            ["llama3.2", "granite3.3"],
            "gemma3"
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("llama3.2", results)
        self.assertIn("granite3.3", results)
        self.assertEqual(results["llama3.2"]["output"], self.test_output)
        self.assertEqual(results["granite3.3"]["output"], self.test_output)
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)
        
        # Verify API calls
        self.assertTrue(mock_get.called)
        self.assertTrue(mock_post.called)

    @patch('requests.post')
    @patch('requests.get')
    def test_lmstudio_translation(self, mock_get, mock_post):
        """Test translation with LMStudio models"""
        # Mock the API responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [
                {"id": "llama3.2"},
                {"id": "granite3.3"},
                {"id": "gemma3"}
            ]
        }
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"text": self.test_output}]
        }
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="lmstudio",
            reference_text=self.test_reference,
            project_dir=self.test_dir,
            task_type=TASK_TRANSLATION,
            source_language="English",
            target_language="Spanish"
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            self.test_content,
            ["llama3.2", "granite3.3"],
            "gemma3"
        )
        
        # Assertions
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("llama3.2", results)
        self.assertIn("granite3.3", results)
        self.assertEqual(results["llama3.2"]["output"], self.test_output)
        self.assertEqual(results["granite3.3"]["output"], self.test_output)
        
        # Check that files were created in the project directory
        files = os.listdir(self.test_dir)
        self.assertGreater(len(files), 0)
        
        # Verify API calls
        self.assertTrue(mock_get.called)
        self.assertTrue(mock_post.called)

    @patch('requests.post')
    def test_rouge_evaluation(self, mock_post):
        """Test ROUGE evaluation"""
        # Skip if rouge is not available
        try:
            from rouge import Rouge
        except ImportError:
            self.skipTest("rouge package not available")
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=self.test_reference,
            task_type=TASK_SUMMARIZATION
        )
        
        # Test the ROUGE evaluation
        rouge_scores = evaluator.evaluate_with_rouge(self.test_output)
        
        # Assertions
        self.assertIsInstance(rouge_scores, dict)
        self.assertIn("rouge-1 (F-Score)", rouge_scores)
        self.assertIn("rouge-2 (F-Score)", rouge_scores)
        self.assertIn("rouge-l (F-Score)", rouge_scores)

    @patch('requests.post')
    def test_bert_score_evaluation(self, mock_post):
        """Test BERTScore evaluation"""
        # Skip if bert_score is not available
        try:
            from bert_score import BERTScorer
        except ImportError:
            self.skipTest("bert_score package not available")
        
        # Create evaluator instance with a mock BERTScorer
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=self.test_reference,
            task_type=TASK_SUMMARIZATION
        )
        
        # Mock the BERTScorer
        evaluator.bert_scorer = MagicMock()
        evaluator.bert_scorer.score.return_value = (
            pd.Series([0.8]), pd.Series([0.7]), pd.Series([0.75])
        )
        
        # Test the BERTScore evaluation
        bert_scores = evaluator.evaluate_with_bert_score(self.test_output)
        
        # Assertions
        self.assertIsInstance(bert_scores, dict)
        self.assertIn("BERTScore (F1)", bert_scores)

    @patch('requests.post')
    def test_bleu_evaluation(self, mock_post):
        """Test BLEU evaluation"""
        # Skip if sacrebleu is not available
        try:
            import sacrebleu
        except ImportError:
            self.skipTest("sacrebleu package not available")
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            reference_text=self.test_reference,
            task_type=TASK_TRANSLATION
        )
        
        # Test the BLEU evaluation
        bleu_scores = evaluator.evaluate_with_bleu(self.test_output)
        
        # Assertions
        self.assertIsInstance(bleu_scores, dict)
        self.assertIn("BLEU", bleu_scores)

    @patch('requests.post')
    def test_llm_evaluation(self, mock_post):
        """Test LLM-based evaluation"""
        # Mock the API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "4"  # Simple response with just a score
        }
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            task_type=TASK_SUMMARIZATION
        )
        
        # Test the LLM evaluation
        llm_scores = evaluator.evaluate_with_llm(
            "gemma3:latest", 
            self.test_content, 
            self.test_output
        )
        
        # Assertions
        self.assertIsInstance(llm_scores, dict)
        self.assertGreater(len(llm_scores), 0)
        
        # Verify API calls
        self.assertTrue(mock_post.called)

    def test_generate_evaluation_report(self):
        """Test report generation"""
        # Create evaluator instance
        evaluator = ModelEvaluator(
            platform="ollama",
            project_dir=self.test_dir,
            task_type=TASK_SUMMARIZATION
        )
        
        # Create sample results
        results = {
            "llama3.2:latest": {
                "output": "Sample summary 1",
                "metrics": {
                    "Relevance": 4,
                    "Coherence": 5,
                    "Consistency": 4,
                    "Fluency": 3
                }
            },
            "granite3.3:latest": {
                "output": "Sample summary 2",
                "metrics": {
                    "Relevance": 5,
                    "Coherence": 4,
                    "Consistency": 5,
                    "Fluency": 3
                }
            }
        }
        
        # Generate report
        timestamp = "20250511_000000"
        evaluator.generate_evaluation_report(results, timestamp)
        
        # Check that report file was created
        report_path = os.path.join(self.test_dir, f"{timestamp}_evaluation_report.md")
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        self.assertIn("Summarization Evaluation Report", report_content)
        self.assertIn("Executive Summary", report_content)
        self.assertIn("Detailed Results", report_content)
        self.assertIn("llama3.2:latest", report_content)
        self.assertIn("granite3.3:latest", report_content)

if __name__ == '__main__':
    unittest.main()
