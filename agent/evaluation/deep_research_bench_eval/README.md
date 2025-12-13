## Introduction

We conduct external evaluation on Deep Research Bench (DRB) with [their original repository](https://github.com/Ayanami0730/deep_research_bench) to aquire the evaluation results for two different metrics: 1. Reference-based and Adaptive Criteria-driven Evaluation framework with Dynamic Weighting (RACE) and 2. Factual Abundance and Citation Trustworthiness (FACT)

We provide our format conversion script and step-by-step instructions as follows.


## Steps

### 1. Generate the DRB output from our system
Our repository supports the generation natively. Add `deep_research_bench` to the task in the scripts, e.g., `agent/scripts/auto_search.sh`, to acquire the output. The default path will be `eval_output/auto_search_sft/deep_research_bench-ablation-s2.jsonl`

### 2. Conduct format conversion and place the files into the original repository
In the original DRB evaluation, the authors use [JINA API](https://jina.ai/) to scrape the URLs in the generated deep research reports. In our pipeline, since we already acquire the URL content through our search and browering, we directly use the scraped content to pair with the sentences in the article for citation evaluation.

Our format conversion code is in `drb_formatter.py`. It requires three positional arguments: 
- input_file_path: the path to the output file from our system.
- task_name: the identifier of this task, e.g., deep_research_bench-ablation-s2.
- drb_repo_path: the path to the official DRB repo (details in Step 3), where the formatted data will be stored.

A full example:
```bash
python drb_formatter.py \
    --input_file_path /path/to/drb-ablation-s2.jsonl \
    --task_name drb-abaltion-s2 \
    --drb_repo_path /path/to/deep_research_bench
```

### 3. Step up the original GitHub Repo

#### 3.1. Prerequisites

- Python 3.9+
- Gemini API key (for LLM evaluation)
- Jina API key (for web scraping in FACT evaluation)

#### 3.2. Set up and create a new environment

```bash
git clone https://github.com/Ayanami0730/deep_research_bench
cd deep_research_bench
conda create -n drb python=3.9
conda activate drb
pip install -r requirements.txt
```

🚨 **Crucial**: The original citation evaluation model is deprecated, go to `/path/to/deep_research_bench/utils/api.py` of the repo and change the `FACT_Moedel` to `gemini-2.5-flash-preview-09-2025` to avoid errors in citation evaluation.

#### 3.3 API Configuration

Set the required API keys as environment variables:

```bash
# Set Gemini API key for LLM evaluation
export GEMINI_API_KEY="your_gemini_api_key_here"

# Set Jina API key for web scraping
export JINA_API_KEY="your_jina_api_key_here"
```

#### 1.4 Quick format guide (in the DRB repo)
- Original query file: `data/prompt_data/query.jsonl`
- Example output file: `data/test_data/raw_data/claude-3-7-sonnet-latest.jsonl`


### 4. Run the evaluation and acquire the results

#### 4.1. Configure the task to evaluate
- After the setup of the original repository, copy our eval script (`run_benchmark_scraped.sh`) in this folder to the root directory of repository folder

- Edit `run_benchmark_scraped.sh` and add your `task_name` (as you specified in Step 2):

```bash
TARGET_MODELS=("drb-abaltion-s2")
```

#### 4.2. Run the evaluation
Run the evaluation under the DRB repo folder and the corresponding environment.
```bash
bash run_benchmark_scraped.sh
```
The evaluation will take around 40 minutes. You can check the progress in `output_$task_name.log`.

#### 4.3. Check the results
By default, your RACE scores (for the article) will be scored in `output_$task_name.log` in the root directory of repository folder. Your FACT scores (for the citations) will be scored in `results/fact/$task_name/fact_result.txt` under the repository folder