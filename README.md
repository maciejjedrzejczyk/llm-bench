# LLM-Bench: Local Model Evaluation Framework

LLM-Bench is a comprehensive framework for evaluating local LLM models (from Ollama or LMStudio) on summarization and translation tasks using various evaluation metrics. It's inspired by the evaluation techniques demonstrated in the `evaluate.ipynb` notebook.

## Features

- Evaluate multiple local models on summarization and translation tasks
- Input content from a file or URL
- Support for both Ollama and LMStudio platforms
- Multiple evaluation metrics:
  - ROUGE (if reference summary is provided)
  - BERTScore (if reference summary is provided)
  - LLM-based evaluation (using another local model as evaluator)
- Interactive and command-line modes
- Save evaluation results to JSON

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/evaluate.git
cd evaluate
```

2. Create and activate a virtual environment (recommended):
```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install all dependencies using the requirements file:
```bash
pip install -r requirements.txt
```

Note: If you don't need all evaluation metrics, you can install just the core dependencies:
```bash
pip install requests beautifulsoup4 pandas numpy
```

## Usage

### Interactive Mode

Run the script in interactive mode to be guided through the evaluation process:

```bash
python model_evaluator.py -i
```

### Command-Line Mode

```bash
# For summarization task
python model_evaluator.py -t summarization -p ollama -s input.txt -m llama2,mistral -e llama2 -o results.json

# For translation task
python model_evaluator.py -t translation -p ollama -s input.txt -m llama2,mistral --source-lang English --target-lang Spanish -r reference.txt -o results.json
```

### Command-Line Arguments

- `-i, --interactive`: Run in interactive mode
- `-t, --task`: Task type (summarization or translation)
- `-p, --platform`: Platform to use (ollama or lmstudio)
- `-s, --source`: Content source (file path or URL)
- `-m, --models`: Comma-separated list of models to evaluate
- `-r, --reference`: Reference text for comparison
- `-e, --evaluator`: Model to use for LLM-based evaluation
- `-o, --output`: Output file for results
- `-d, --project-dir`: Project directory for saving all evaluation files
- `--source-lang`: Source language for translation task
- `--target-lang`: Target language for translation task

## Evaluation Metrics

### For Summarization

#### ROUGE
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap of n-grams between the generated summary and a reference summary. It's useful for evaluating how well the generated summary captures the content of the reference.

#### BERTScore
BERTScore leverages contextual embeddings from BERT to evaluate the semantic similarity between generated and reference summaries. It can capture semantic similarities that might be missed by n-gram based metrics like ROUGE.

### For Translation

#### BLEU
BLEU (Bilingual Evaluation Understudy) is a precision-focused metric that measures how many of the n-grams in the generated translation appear in the reference translation. It includes a brevity penalty to prevent very short translations from scoring too high.

#### BERTScore
BERTScore can also be used for translation evaluation, measuring the semantic similarity between the generated translation and the reference.

### For Both Tasks

#### LLM-based Evaluation
This approach uses another LLM to evaluate the quality of generated outputs based on criteria like:
- Relevance: How well the output includes important information from the source
- Coherence: How well-structured and organized the output is
- Consistency: The factual alignment between the output and the source
- Fluency: The quality of grammar, spelling, and sentence structure

## Example

```bash
# Evaluate llama2 and mistral models on a news article
python model_evaluator.py -p ollama -s https://example.com/article.html -m llama2,mistral -e llama2 -o results.json

# Evaluate with a reference summary
python model_evaluator.py -p ollama -s input.txt -m llama2,mistral -r reference.txt -o results.json

# Save all evaluation files to a project directory
python model_evaluator.py -p ollama -s input.txt -m llama2,mistral -e llama2 -d ./evaluations/run1
```

## Project Directory Structure

When using the `-d` or `--project-dir` option, the script creates a structured evaluation project with the following files:

```
project_dir/
├── YYYYMMDD_HHMMSS_config.json         # Configuration details
├── YYYYMMDD_HHMMSS_content.txt         # Source content
├── YYYYMMDD_HHMMSS_reference.txt       # Reference summary (if provided)
├── YYYYMMDD_HHMMSS_model1_summary.txt  # Summary from model1
├── YYYYMMDD_HHMMSS_model2_summary.txt  # Summary from model2
├── YYYYMMDD_HHMMSS_model1_evaluation.json  # Evaluation results for model1
├── YYYYMMDD_HHMMSS_model2_evaluation.json  # Evaluation results for model2
└── YYYYMMDD_HHMMSS_results.json        # Complete evaluation results
```

This structure makes it easy to organize and compare multiple evaluation runs.

## Limitations

- The quality of LLM-based evaluation depends on the capabilities of the evaluator model
- ROUGE and BERTScore require a reference summary, which may not always be available
- Local models may have context length limitations that affect summarization of long texts
- The script assumes that the Ollama API is running on localhost:11434 and LMStudio on localhost:1234
