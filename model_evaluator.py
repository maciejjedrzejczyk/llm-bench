#!/usr/bin/env python3
"""
Model Evaluator

This script evaluates local LLM models (from Ollama or LMStudio) on summarization or translation tasks.
It allows users to input content from a file or URL, then evaluates multiple models
using various evaluation metrics.
"""

import argparse
import os
import sys
import re
import json
import time
import requests
import datetime
import shutil
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import numpy as np

# Try to import evaluation libraries, with graceful fallbacks
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge package not found. ROUGE metrics will not be available.")
    print("Install with: pip install rouge")

try:
    from bert_score import BERTScorer
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert_score package not found. BERTScore metrics will not be available.")
    print("Install with: pip install bert_score")

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu package not found. BLEU metrics will not be available.")
    print("Install with: pip install sacrebleu")

# Constants
OLLAMA_API_BASE = "http://localhost:11434/api"
LMSTUDIO_API_BASE = "http://localhost:1234/v1"

# Task types
TASK_SUMMARIZATION = "summarization"
TASK_TRANSLATION = "translation"
# Evaluation prompt templates based on G-Eval
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form:
- {metric_name}: [PROVIDE ONLY A SINGLE DIGIT BETWEEN 1-5]

Your response should contain ONLY the digit representing your score, nothing else.
"""

# Metric 1: Relevance
RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""

RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

# Metric 2: Coherence
COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 3: Consistency
CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the summarized source. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""

CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.
"""

# Metric 4: Fluency
FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""

# Summarization prompt template
SUMMARIZATION_PROMPT = """
Please provide a concise summary of the following text. Focus on capturing the main points and key information:

{text}

Summary:
"""

# Translation prompt template
TRANSLATION_PROMPT = """
Please translate the following text from {source_language} to {target_language}:

{text}

Translation:
"""
class ModelEvaluator:
    """Class to evaluate local LLM models on summarization or translation tasks"""
    
    def __init__(self, platform: str = "ollama", reference_text: Optional[str] = None, 
                 project_dir: Optional[str] = None, task_type: str = TASK_SUMMARIZATION,
                 source_language: str = "English", target_language: str = "Spanish"):
        """
        Initialize the evaluator
        
        Args:
            platform: The platform to use ("ollama" or "lmstudio")
            reference_text: Optional reference summary or translation for comparison
            project_dir: Optional project directory for saving all evaluation files
            task_type: Type of task to evaluate ("summarization" or "translation")
            source_language: Source language for translation task
            target_language: Target language for translation task
        """
        self.platform = platform.lower()
        self.reference_text = reference_text
        self.api_base = OLLAMA_API_BASE if platform.lower() == "ollama" else LMSTUDIO_API_BASE
        self.project_dir = project_dir
        self.task_type = task_type
        self.source_language = source_language
        self.target_language = target_language
        
        # Create project directory if specified and doesn't exist
        if self.project_dir and not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        
        # Initialize evaluation metrics
        self.evaluation_metrics = {
            "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
            "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
            "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
            "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
        }
        
        # Initialize ROUGE if available
        if ROUGE_AVAILABLE:
            self.rouge = Rouge()
        
        # Initialize BERTScorer if available
        if BERT_SCORE_AVAILABLE:
            self.bert_scorer = BERTScorer(lang="en")
    def list_available_models(self) -> List[str]:
        """List available models on the platform"""
        try:
            if self.platform == "ollama":
                response = requests.get(f"{self.api_base}/tags")
                if response.status_code == 200:
                    models = [model["name"] for model in response.json()["models"]]
                    return models
                else:
                    print(f"Error listing models: {response.status_code}")
                    return []
            else:  # LMStudio
                # Use the LMStudio API to list models
                response = requests.get("http://127.0.0.1:1234/v1/models")
                if response.status_code == 200:
                    models_data = response.json()
                    if "data" in models_data:
                        models = [model["id"] for model in models_data["data"]]
                        return models
                    else:
                        print("Unexpected response format from LMStudio API")
                        return []
                else:
                    print(f"Error listing models from LMStudio API: {response.status_code}")
                    print("Please ensure LMStudio is running and the API is accessible.")
                    return []
        except Exception as e:
            print(f"Error listing models: {e}")
            print("Please ensure the platform is running and accessible.")
            return []
    
    def extract_content_from_url(self, url: str) -> str:
        """Extract content from a URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"Error extracting content from URL: {e}")
            return ""
    
    def read_content_from_file(self, file_path: str) -> str:
        """Read content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def get_content(self, source: str) -> str:
        """Get content from a file or URL"""
        if urlparse(source).scheme in ('http', 'https'):
            return self.extract_content_from_url(source)
        else:
            return self.read_content_from_file(source)
    def generate_output(self, model: str, text: str) -> str:
        """Generate output (summary or translation) using the specified model"""
        if self.task_type == TASK_SUMMARIZATION:
            prompt = SUMMARIZATION_PROMPT.format(text=text)
        else:  # Translation
            prompt = TRANSLATION_PROMPT.format(
                source_language=self.source_language,
                target_language=self.target_language,
                text=text
            )
        
        try:
            if self.platform == "ollama":
                response = requests.post(
                    f"{self.api_base}/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    return response.json()["response"]
                else:
                    print(f"Error generating output with {model}: {response.status_code}")
                    return ""
            else:  # LMStudio
                response = requests.post(
                    f"{self.api_base}/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["text"]
                else:
                    print(f"Error generating output with {model}: {response.status_code}")
                    return ""
        except Exception as e:
            print(f"Error generating output with {model}: {e}")
            return ""
    
    def evaluate_with_rouge(self, text: str) -> Dict[str, float]:
        """Evaluate a text using ROUGE metrics"""
        if not ROUGE_AVAILABLE or not self.reference_text:
            return {}
        
        try:
            scores = self.rouge.get_scores(text, self.reference_text)[0]
            return {
                "rouge-1 (F-Score)": scores["rouge-1"]["f"],
                "rouge-2 (F-Score)": scores["rouge-2"]["f"],
                "rouge-l (F-Score)": scores["rouge-l"]["f"]
            }
        except Exception as e:
            print(f"Error evaluating with ROUGE: {e}")
            return {}
    
    def evaluate_with_bert_score(self, text: str) -> Dict[str, float]:
        """Evaluate a text using BERTScore"""
        if not BERT_SCORE_AVAILABLE or not self.reference_text:
            return {}
        
        try:
            P, R, F1 = self.bert_scorer.score([text], [self.reference_text])
            return {"BERTScore (F1)": F1.item()}
        except Exception as e:
            print(f"Error evaluating with BERTScore: {e}")
            return {}
    
    def evaluate_with_bleu(self, text: str) -> Dict[str, float]:
        """Evaluate a translation using BLEU score"""
        if not BLEU_AVAILABLE or not self.reference_text:
            return {}
        
        try:
            bleu = sacrebleu.corpus_bleu([text], [[self.reference_text]])
            return {"BLEU": bleu.score}
        except Exception as e:
            print(f"Error evaluating with BLEU: {e}")
            return {}
    def evaluate_with_llm(self, model: str, document: str, summary: str) -> Dict[str, int]:
        """Evaluate a summary using another LLM as evaluator"""
        results = {}
        
        for metric_name, (criteria, steps) in self.evaluation_metrics.items():
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                criteria=criteria,
                steps=steps,
                document=document,
                summary=summary,
                metric_name=metric_name
            )
            
            try:
                if self.platform == "ollama":
                    response = requests.post(
                        f"{self.api_base}/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()["response"].strip()
                        # Debug output to see the actual response
                        print(f"DEBUG - {metric_name} response: '{result}'")
                        
                        # Check for empty responses
                        if not result:
                            print(f"Warning: Empty response for {metric_name}")
                            continue
                            
                        # Extract the numeric score with improved flexible regex pattern
                        if metric_name == "Fluency":
                            score_match = re.search(r'(?:^|.*?)(?:score|rating|is|[:=]|[-–—])\s*(?:a\s*)?([1-3])(?:\s|$|\.|\,|\/|\)|;|:|$)', result, re.IGNORECASE)
                        else:
                            score_match = re.search(r'(?:^|.*?)(?:score|rating|is|[:=]|[-–—])\s*(?:a\s*)?([1-5])(?:\s|$|\.|\,|\/|\)|;|:|$)', result, re.IGNORECASE)
                        
                        if score_match:
                            results[metric_name] = int(score_match.group(1))
                        else:
                            # Try a simpler approach - just find any digit between 1-5 (or 1-3 for fluency)
                            if metric_name == "Fluency":
                                digits = re.findall(r'[1-3]', result)
                            else:
                                digits = re.findall(r'[1-5]', result)
                            
                            if digits:
                                # Use the first valid digit found
                                results[metric_name] = int(digits[0])
                            else:
                                print(f"Warning: Could not extract valid score for {metric_name}")
                else:  # LMStudio
                    response = requests.post(
                        f"{self.api_base}/completions",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": 10,
                            "temperature": 0
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()["choices"][0]["text"].strip()
                        # Debug output to see the actual response
                        print(f"DEBUG - {metric_name} response: '{result}'")
                        
                        # Check for empty responses
                        if not result:
                            print(f"Warning: Empty response for {metric_name}")
                            continue
                            
                        # Extract the numeric score with improved flexible regex pattern
                        if metric_name == "Fluency":
                            score_match = re.search(r'(?:^|.*?)(?:score|rating|is|[:=]|[-–—])\s*(?:a\s*)?([1-3])(?:\s|$|\.|\,|\/|\)|;|:|$)', result, re.IGNORECASE)
                        else:
                            score_match = re.search(r'(?:^|.*?)(?:score|rating|is|[:=]|[-–—])\s*(?:a\s*)?([1-5])(?:\s|$|\.|\,|\/|\)|;|:|$)', result, re.IGNORECASE)
                        
                        if score_match:
                            results[metric_name] = int(score_match.group(1))
                        else:
                            # Try a simpler approach - just find any digit between 1-5 (or 1-3 for fluency)
                            if metric_name == "Fluency":
                                digits = re.findall(r'[1-3]', result)
                            else:
                                digits = re.findall(r'[1-5]', result)
                            
                            if digits:
                                # Use the first valid digit found
                                results[metric_name] = int(digits[0])
                            else:
                                print(f"Warning: Could not extract valid score for {metric_name}")
            except Exception as e:
                print(f"Error evaluating {metric_name} with LLM: {e}")
        
        return results
    def generate_evaluation_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """Generate a markdown report comparing model performance"""
        if not results:
            return
        
        # Check if there are any metrics to report on
        if not any(model_results.get("metrics", {}) for model_results in results.values()):
            print("No evaluation metrics available to generate report")
            return
        
        task_name = "summarization" if self.task_type == TASK_SUMMARIZATION else "translation"
        
        # Extract metrics for all models
        metrics_data = []
        for model, model_results in results.items():
            row = {"Model": model}
            row.update(model_results.get("metrics", {}))
            metrics_data.append(row)
        
        if not metrics_data or not any(len(row) > 1 for row in metrics_data):
            print("No metrics to report")
            return  # No metrics to report
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Calculate rankings for each metric
        ranking_df = metrics_df.copy()
        metric_columns = [col for col in ranking_df.columns if col != "Model"]
        
        rank_columns = []
        for column in metric_columns:
            # Skip columns with missing values
            if ranking_df[column].isna().any():
                print(f"Warning: Skipping ranking for {column} due to missing values")
                continue
            
            try:
                # Rank the models (higher is better)
                ranking_df[f"{column} Rank"] = ranking_df[column].rank(ascending=False).astype(int)
                rank_columns.append(f"{column} Rank")
            except Exception as e:
                print(f"Warning: Could not rank for {column}: {e}")
        
        # Calculate overall rank based on average of individual ranks (if any)
        if rank_columns:
            try:
                ranking_df["Overall Rank"] = ranking_df[rank_columns].mean(axis=1).round(2)
                # Sort by overall rank
                ranking_df = ranking_df.sort_values("Overall Rank")
            except Exception as e:
                print(f"Warning: Could not calculate overall rank: {e}")
        
        # Find best model overall and for each metric
        best_overall = None
        if "Overall Rank" in ranking_df.columns and not ranking_df.empty:
            best_overall = ranking_df.iloc[0]["Model"]
            
        best_by_metric = {}
        for column in metric_columns:
            if column in metrics_df.columns and not metrics_df[column].isna().all():
                try:
                    best_idx = metrics_df[column].idxmax()
                    if best_idx is not None:
                        best_by_metric[column] = metrics_df.loc[best_idx, "Model"]
                except Exception as e:
                    print(f"Warning: Could not determine best model for {column}: {e}")
        
        # Generate markdown report
        report = f"# {task_name.capitalize()} Evaluation Report\n\n"
        report += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive summary
        report += "## Executive Summary\n\n"
        if best_overall:
            report += f"**Best Overall Model**: {best_overall}\n\n"
        
        if best_by_metric:
            report += "**Best Model by Metric**:\n"
            for metric, model in best_by_metric.items():
                report += f"- {metric}: {model}\n"
        else:
            report += "No complete metrics available for ranking models.\n"
        
        # Detailed results table
        if not metrics_df.empty and len(metrics_df.columns) > 1:
            report += "\n## Detailed Results\n\n"
            report += "| Model |"
            for column in metric_columns:
                if f"{column} Rank" in ranking_df.columns:
                    report += f" {column} | {column} Rank |"
                else:
                    report += f" {column} | |"
            if "Overall Rank" in ranking_df.columns:
                report += " Overall Rank |"
            report += "\n"
            
            # Add separator row
            report += "|" + "---|" * (len(metric_columns) * 2 + 1)
            if "Overall Rank" in ranking_df.columns:
                report += "---|"
            report += "\n"
            
            # Add data rows
            for _, row in ranking_df.iterrows():
                report += f"| {row['Model']} |"
                for column in metric_columns:
                    value = row.get(column)
                    rank = row.get(f"{column} Rank", "")
                    
                    # Format the value properly
                    if pd.isna(value):
                        value_str = "N/A"
                    elif isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                        
                    report += f" {value_str} | {rank} |"
                    
                if "Overall Rank" in ranking_df.columns:
                    report += f" {row.get('Overall Rank', '')} |"
                report += "\n"
        
        # Save report to file if project directory is specified
        if self.project_dir:
            report_path = os.path.join(self.project_dir, f"{timestamp}_evaluation_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Evaluation report saved to {report_path}")
        
        # Print report
        print("\n=== Evaluation Report ===\n")
        print(report)
    def run_evaluation(self, content: str, models: List[str], evaluator_model: Optional[str] = None) -> Dict[str, Any]:
        """Run the full evaluation pipeline"""
        results = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        task_name = "summarization" if self.task_type == TASK_SUMMARIZATION else "translation"
        print(f"Generating {task_name} outputs for {len(models)} models...")
        
        # Save configuration if project directory is specified
        if self.project_dir:
            config = {
                "timestamp": timestamp,
                "platform": self.platform,
                "task_type": self.task_type,
                "models": models,
                "evaluator_model": evaluator_model,
                "has_reference": bool(self.reference_text),
                "content_length": len(content)
            }
            if self.task_type == TASK_TRANSLATION:
                config["source_language"] = self.source_language
                config["target_language"] = self.target_language
                
            config_path = os.path.join(self.project_dir, f"{timestamp}_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {config_path}")
            
            # Save content
            content_path = os.path.join(self.project_dir, f"{timestamp}_content.txt")
            with open(content_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Content saved to {content_path}")
            
            # Save reference text if available
            if self.reference_text:
                ref_path = os.path.join(self.project_dir, f"{timestamp}_reference.txt")
                with open(ref_path, 'w', encoding='utf-8') as f:
                    f.write(self.reference_text)
                print(f"Reference text saved to {ref_path}")
        
        # Generate outputs for each model
        outputs = {}
        for model in models:
            print(f"Generating {task_name} with {model}...")
            output = self.generate_output(model, content)
            if output:
                outputs[model] = output
                print(f"Output generated: {len(output)} characters")
                
                # Save individual output if project directory is specified
                if self.project_dir:
                    output_path = os.path.join(self.project_dir, f"{timestamp}_{model}_{task_name}.txt")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                    print(f"Output saved to {output_path}")
            else:
                print(f"Failed to generate output with {model}")
        
        print(f"Generated {len(outputs)} outputs")
        # Check if any evaluation method is available
        has_evaluation_method = False
        
        if self.task_type == TASK_SUMMARIZATION:
            has_evaluation_method = (ROUGE_AVAILABLE and self.reference_text) or \
                                   (BERT_SCORE_AVAILABLE and self.reference_text) or \
                                   evaluator_model
        else:  # Translation
            has_evaluation_method = (BLEU_AVAILABLE and self.reference_text) or \
                                   (BERT_SCORE_AVAILABLE and self.reference_text) or \
                                   evaluator_model
        
        if not has_evaluation_method:
            print("\nWARNING: No evaluation method available!")
            print("To enable evaluation, you need to either:")
            print(f"  1. Provide a reference {task_name} (for {'ROUGE/BERTScore' if self.task_type == TASK_SUMMARIZATION else 'BLEU/BERTScore'})")
            print("  2. Specify an evaluator model for LLM-based evaluation")
            print(f"{task_name.capitalize()}s have been generated but will not be evaluated.")
        
        # Evaluate outputs
        for model, output in outputs.items():
            results[model] = {
                "output": output,
                "metrics": {}
            }
            
            # Task-specific evaluation
            if self.task_type == TASK_SUMMARIZATION:
                # ROUGE evaluation
                if ROUGE_AVAILABLE and self.reference_text:
                    print(f"Evaluating {model} with ROUGE...")
                    rouge_scores = self.evaluate_with_rouge(output)
                    results[model]["metrics"].update(rouge_scores)
            else:  # Translation
                # BLEU evaluation
                if BLEU_AVAILABLE and self.reference_text:
                    print(f"Evaluating {model} with BLEU...")
                    bleu_scores = self.evaluate_with_bleu(output)
                    results[model]["metrics"].update(bleu_scores)
            
            # BERTScore evaluation (for both tasks)
            if BERT_SCORE_AVAILABLE and self.reference_text:
                print(f"Evaluating {model} with BERTScore...")
                bert_scores = self.evaluate_with_bert_score(output)
                results[model]["metrics"].update(bert_scores)
            
            # LLM-based evaluation
            if evaluator_model:
                print(f"Evaluating {model} with LLM ({evaluator_model})...")
                llm_scores = self.evaluate_with_llm(evaluator_model, content, output)
                results[model]["metrics"].update(llm_scores)
                
                # Save individual evaluation results if project directory is specified
                if self.project_dir and llm_scores:
                    eval_path = os.path.join(self.project_dir, f"{timestamp}_{model}_evaluation.json")
                    with open(eval_path, 'w', encoding='utf-8') as f:
                        json.dump(llm_scores, f, indent=2)
                    print(f"Evaluation results saved to {eval_path}")
        
        # Save complete results if project directory is specified
        if self.project_dir:
            results_path = os.path.join(self.project_dir, f"{timestamp}_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Complete results saved to {results_path}")
        
        # Generate evaluation report
        if has_evaluation_method and results:
            self.generate_evaluation_report(results, timestamp)
        
        return results
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display evaluation results in a formatted table"""
        if not results:
            print("No results to display")
            return
        
        task_name = "summarization" if self.task_type == TASK_SUMMARIZATION else "translation"
        
        # Extract all unique metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results["metrics"].keys())
        
        # Create DataFrame for metrics
        metrics_data = []
        for model, model_results in results.items():
            row = {"Model": model}
            for metric in all_metrics:
                row[metric] = model_results["metrics"].get(metric, None)
            metrics_data.append(row)
        
        if metrics_data:
            if all_metrics:
                metrics_df = pd.DataFrame(metrics_data)
                print("\n=== Evaluation Metrics ===")
                print(metrics_df.to_string(index=False))
                
                # Generate markdown table with rankings
                self.generate_ranking_table(metrics_df)
            else:
                print("\n=== No Evaluation Metrics Available ===")
                print("No metrics were calculated because no evaluation method was enabled.")
                print("To enable evaluation, you need to either:")
                print(f"  1. Provide a reference {task_name} (for {'ROUGE/BERTScore' if self.task_type == TASK_SUMMARIZATION else 'BLEU/BERTScore'})")
                print("  2. Specify an evaluator model for LLM-based evaluation")
        
        # Print outputs
        print(f"\n=== Generated {task_name.capitalize()}s ===")
        for model, model_results in results.items():
            print(f"\n--- {model} ---")
            print(model_results["output"])
    
    def generate_ranking_table(self, metrics_df: pd.DataFrame) -> None:
        """Generate a markdown table with model rankings for each metric"""
        if len(metrics_df) <= 1:
            return  # No need for rankings with only one model
        
        # Create a copy of the DataFrame for rankings
        ranking_df = metrics_df.copy()
        
        # Calculate rankings for each metric
        for column in ranking_df.columns:
            if column != "Model":
                # Skip columns with missing values
                if ranking_df[column].isna().any():
                    continue
                
                # Rank the models (higher is better)
                ranking_df[f"{column} Rank"] = ranking_df[column].rank(ascending=False).astype(int)
        
        # Calculate overall rank based on average of individual ranks
        rank_columns = [col for col in ranking_df.columns if col.endswith(" Rank")]
        if rank_columns:
            ranking_df["Overall Rank"] = ranking_df[rank_columns].mean(axis=1).round(2)
            # Sort by overall rank
            ranking_df = ranking_df.sort_values("Overall Rank")
        
        # Create markdown table
        markdown_table = "| Model |"
        for column in metrics_df.columns:
            if column != "Model":
                markdown_table += f" {column} | {column} Rank |"
        if "Overall Rank" in ranking_df.columns:
            markdown_table += " Overall Rank |"
        markdown_table += "\n"
        
        # Add separator row
        markdown_table += "|" + "---|" * (len(metrics_df.columns) * 2)
        if "Overall Rank" in ranking_df.columns:
            markdown_table += "---|"
        markdown_table += "\n"
        
        # Add data rows
        for _, row in ranking_df.iterrows():
            markdown_table += f"| {row['Model']} |"
            for column in metrics_df.columns:
                if column != "Model":
                    value = row[column]
                    rank = row.get(f"{column} Rank", "N/A")
                    markdown_table += f" {value} | {rank} |"
            if "Overall Rank" in ranking_df.columns:
                markdown_table += f" {row['Overall Rank']} |"
            markdown_table += "\n"
        
        print("\n=== Model Rankings (Markdown Table) ===")
        print(markdown_table)
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save results to a JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
def interactive_mode():
    """Run the evaluator in interactive mode"""
    print("=== Model Evaluator - Interactive Mode ===")
    
    # Select task type
    task_type = input("Select task type (summarization/translation) [summarization]: ").strip().lower() or TASK_SUMMARIZATION
    if task_type not in [TASK_SUMMARIZATION, TASK_TRANSLATION]:
        print(f"Invalid task type: {task_type}. Using summarization.")
        task_type = TASK_SUMMARIZATION
    
    # Select platform
    platform = input("Select platform (ollama/lmstudio) [ollama]: ").strip() or "ollama"
    
    # Get project directory (optional)
    project_dir = input("Enter project directory for saving all files (optional, press Enter to skip): ").strip()
    
    # Get translation languages if task is translation
    source_language = "English"
    target_language = "Spanish"
    if task_type == TASK_TRANSLATION:
        source_language = input("Enter source language [English]: ").strip() or "English"
        target_language = input("Enter target language [Spanish]: ").strip() or "Spanish"
    
    evaluator = ModelEvaluator(
        platform=platform, 
        project_dir=project_dir, 
        task_type=task_type,
        source_language=source_language,
        target_language=target_language
    )
    
    # Get content source
    content_source = input("Enter content source (file path or URL): ").strip()
    if not content_source:
        print("Error: Content source is required")
        return
    
    content = evaluator.get_content(content_source)
    if not content:
        print("Error: Failed to get content")
        return
    
    print(f"Content loaded: {len(content)} characters")
    
    # Get available models for potential reference text generation
    available_models = evaluator.list_available_models()
    
    # Ask if user wants to generate a reference text
    generate_ref = input(f"Do you want to generate a reference {task_type} first? (y/n) [n]: ").strip().lower() or "n"
    
    if generate_ref == "y":
        if available_models:
            print(f"\nAvailable models for reference {task_type} generation:")
            for i, model in enumerate(available_models):
                print(f"{i+1}. {model}")
            
            ref_model_idx = input(f"Enter model number to use for reference {task_type}: ").strip()
            try:
                idx = int(ref_model_idx) - 1
                if 0 <= idx < len(available_models):
                    ref_model = available_models[idx]
                    print(f"\nGenerating reference {task_type} with {ref_model}...")
                    ref_text = evaluator.generate_output(ref_model, content)
                    if ref_text:
                        print(f"Reference {task_type} generated: {len(ref_text)} characters")
                        print(f"\nReference {task_type.capitalize()}:")
                        print(ref_text)
                        
                        # Ask if user wants to edit the reference text
                        edit_ref = input(f"\nDo you want to edit this reference {task_type}? (y/n) [n]: ").strip().lower() or "n"
                        if edit_ref == "y":
                            print(f"\nEnter your edited reference {task_type} (press Enter twice when done):")
                            lines = []
                            while True:
                                line = input()
                                if not line and lines and not lines[-1]:
                                    lines.pop()  # Remove the last empty line
                                    break
                                lines.append(line)
                            ref_text = "\n".join(lines)
                            print(f"Reference {task_type} updated: {len(ref_text)} characters")
                        
                        evaluator.reference_text = ref_text
                    else:
                        print(f"Failed to generate reference {task_type}")
                else:
                    print("Invalid model selection")
            except:
                print(f"Invalid selection. Reference {task_type} generation will be skipped.")
        else:
            ref_model = input(f"Enter model name for reference {task_type} generation: ").strip()
            if ref_model:
                print(f"\nGenerating reference {task_type} with {ref_model}...")
                ref_text = evaluator.generate_output(ref_model, content)
                if ref_text:
                    print(f"Reference {task_type} generated: {len(ref_text)} characters")
                    print(f"\nReference {task_type.capitalize()}:")
                    print(ref_text)
                    
                    # Ask if user wants to edit the reference text
                    edit_ref = input(f"\nDo you want to edit this reference {task_type}? (y/n) [n]: ").strip().lower() or "n"
                    if edit_ref == "y":
                        print(f"\nEnter your edited reference {task_type} (press Enter twice when done):")
                        lines = []
                        while True:
                            line = input()
                            if not line and lines and not lines[-1]:
                                lines.pop()  # Remove the last empty line
                                break
                            lines.append(line)
                        ref_text = "\n".join(lines)
                        print(f"Reference {task_type} updated: {len(ref_text)} characters")
                    
                    evaluator.reference_text = ref_text
                else:
                    print(f"Failed to generate reference {task_type}")
    else:
        # Get reference text (optional) the traditional way
        ref_text = input(f"Enter reference {task_type} (optional, press Enter to skip): ").strip()
        if ref_text:
            evaluator.reference_text = ref_text
    # Get models for evaluation
    if available_models:
        print("\nAvailable models for evaluation:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        selected_indices = input("\nEnter model numbers to evaluate (comma-separated, e.g., 1,3,5): ").strip()
        try:
            indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
            models = [available_models[idx] for idx in indices if 0 <= idx < len(available_models)]
        except:
            print("Invalid selection. Please enter comma-separated numbers.")
            return
    else:
        models_input = input("Enter model names (comma-separated): ").strip()
        models = [model.strip() for model in models_input.split(",") if model.strip()]
    
    if not models:
        print("Error: No models selected")
        return
    
    # Get evaluator model (optional)
    evaluator_model = None
    use_llm_eval = input("Use LLM-based evaluation? (y/n) [y]: ").strip().lower() or "y"
    if use_llm_eval == "y":
        if available_models:
            print("\nSelect evaluator model:")
            for i, model in enumerate(available_models):
                print(f"{i+1}. {model}")
            
            eval_idx = input("Enter evaluator model number: ").strip()
            try:
                idx = int(eval_idx) - 1
                if 0 <= idx < len(available_models):
                    evaluator_model = available_models[idx]
            except:
                print("Invalid selection. LLM evaluation will be skipped.")
        else:
            evaluator_model = input("Enter evaluator model name: ").strip()
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.run_evaluation(content, models, evaluator_model)
    
    # Display results
    evaluator.display_results(results)
    
    # Save results if no project directory was specified
    if not project_dir:
        save_results = input("\nSave results to file? (y/n) [y]: ").strip().lower() or "y"
        if save_results == "y":
            output_file = input("Enter output file path [results.json]: ").strip() or "results.json"
            evaluator.save_results(results, output_file)
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Evaluate local LLM models on summarization or translation tasks")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-t", "--task", choices=[TASK_SUMMARIZATION, TASK_TRANSLATION], default=TASK_SUMMARIZATION, 
                        help="Task type (summarization or translation)")
    parser.add_argument("-p", "--platform", choices=["ollama", "lmstudio"], default="ollama", help="Platform to use")
    parser.add_argument("-s", "--source", help="Content source (file path or URL)")
    parser.add_argument("-m", "--models", help="Comma-separated list of models to evaluate")
    parser.add_argument("-r", "--reference", help="Reference text for comparison")
    parser.add_argument("-e", "--evaluator", help="Model to use for LLM-based evaluation")
    parser.add_argument("-o", "--output", default="results.json", help="Output file for results")
    parser.add_argument("-d", "--project-dir", help="Project directory for saving all evaluation files")
    parser.add_argument("--source-lang", default="English", help="Source language for translation task")
    parser.add_argument("--target-lang", default="Spanish", help="Target language for translation task")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    # Non-interactive mode requires source and models
    if not args.source or not args.models:
        parser.print_help()
        return
    
    evaluator = ModelEvaluator(
        platform=args.platform, 
        reference_text=args.reference, 
        project_dir=args.project_dir,
        task_type=args.task,
        source_language=args.source_lang,
        target_language=args.target_lang
    )
    
    content = evaluator.get_content(args.source)
    
    if not content:
        print("Error: Failed to get content")
        return
    
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    
    results = evaluator.run_evaluation(content, models, args.evaluator)
    evaluator.display_results(results)
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
