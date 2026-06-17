# _Monitoring the Internal Monologue: Probe Trajectories Reveal Behavioral Dynamics in Chain-of-Thought_

This is the official repository for the paper: [**_Monitoring the Internal Monologue: Probe Trajectories Reveal Behavioral Dynamics in Chain-of-Thought_**](https://arxiv.org/abs/2605.18549).

## Main Results

Our empirical analysis on predicting future LRM behavior (harmfulness and mathematical correctness) via probe trajectories reveals several key findings:

- **Max-Pooling is Essential:** When aggregating hidden states across the Chain-of-Thought, average-pooling and last-token methods collapse to near-random performance. Max-pooling, however, consistently achieves $>90\%$ AUROC and produces stable probability trajectories.
- **Trajectory Features Improve Prediction:** Treating internal probe predictions as a continuous time-series signal and extracting temporal and signal-processing features yields consistent gains over static probes.
- **Template-Based Training is Sufficient:** High-quality probes can be trained by injecting base dataset samples into static templates. This achieves near-parity with dynamically generated model reasoning traces, eliminating costly data generation steps.
- **Different Behaviors, Different Dynamics:** Safety forecasting is driven by terminal and steady-state features (where the trajectory settles), whereas math error prediction is characterized by volatility and erratic probability spikes.

## Setup

1. **Environment**: It is recommended to use a Python virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Python Path**: Ensure the root of this directory is in your `PYTHONPATH`.
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   ```

## Usage

### 0. Response Generation and Evaluation
Before training or evaluating probes, model responses and safety ground truth must be generated.

**Harmfulness Response Generation (vLLM):**
We use **vLLM** for high-throughput response generation. Use the following script for general datasets:
```bash
python scripts/generation/generate_outputs_for_processed_dataset.py \
    --base_model [MODEL_ID] \
    --dataset_path data/processed_benchmarks/harmfulness/prompt/wildguardtest \
    --save_folder responses/ \
    --output_file model_responses.json
```

**Math Response Generation (lm-eval):**
For math benchmarks (GSM8K, Minerva), we use the **LM Evaluation Harness** with the vLLM backend:
```bash
lm_eval \
  --model vllm \
  --model_args pretrained=[MODEL_ID],dtype=auto,enable_thinking=True \
  --tasks gsm8k,minerva_math \
  --gen_kwargs max_tokens=16384 \
  --log_samples \
  --output_path results/math/eval_results \
  --batch_size auto
```

**Harmfulness Evaluation (WildGuard):**
To establish safety ground truth for generated CoT trajectories, we use the **WildGuard** classifier:
```bash
python scripts/cot_eval/eval_with_wildguard.py \
    --base_model allenai/wildguard \
    --dataset_path data/processed_benchmarks/harmfulness/prompt/wildguardtest \
    --responses_path responses/model_responses.json \
    --save_folder results/harmfulness/eval_wildguard/[MODEL_NAME]
```

### 1. End-to-End Max Probe Training (MIL Probe)
The `train_cot_probe.py` script trains a Multi-Instance Learning (MIL) probe directly on top of a frozen LLM using `lightning` and `hydra`. This is referred to as the **Max Probe**.

**Training with Hydra:**
```bash
python train_cot_probe.py \
    model.base_model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    data.dataset_path=data/processed_benchmarks/harmfulness/prompt/wildguardtrain \
    trainer.trainer_args.devices=1
```
*Note: Configuration is managed via `configs/cot_probe_config/`. You can override any parameter via command line.*

### 2. Max Probe Evaluation
Evaluate a trained MIL probe (Max Probe) on CoT trajectories. This script processes the full sequence and produces per-token and aggregated predictions.

```bash
python scripts/cot_eval/eval_cot_with_mil_probe.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --probe_checkpoint [PATH_TO_CHECKPOINT] \
    --dataset_path data/processed_benchmarks/harmfulness/prompt/wildguardtest \
    --responses_path [COT_RESPONSES_JSON_PATH] \
    --save_folder [OUTPUT_FOLDER]
```

### 3. Probe Trajectory Analysis
Scripts for analyzing the temporal dynamics of probe predictions and training higher-level classifiers on trajectory features are in `scripts/cot_eval/trajectories/`.

**Training Classifiers on Trajectory Features:**
```bash
python scripts/cot_eval/trajectories/train_classifiers.py \
    --model qwen3_8b \
    --experiment_id wildguardtest_response_only \
    --mode harmfulness \
    --output_csv results/classifier_results.csv
```

**Feature Importance Analysis (SHAP):**
```bash
python scripts/cot_eval/trajectories/train_shap_rf.py \
    --model deepseek_r1_distill_llama_8b \
    --experiment_id wildguardtest_scenario_b \
    --mode harmfulness
```

### 4. Two-Stage Probe Training (Average/Last Token)
For efficiency, you can also extract hidden states first and then train a multilayer probe on the static representations.

**Step 1: Hidden State Generation**
```bash
python scripts/harmfulness_concept_extraction/generate_hidden_states.py \
    --base_model [MODEL_ID] \
    --dataset_path [DATASET_PATH] \
    --save_folder [SAVE_PATH] \
    --calculate_mean_hidden_states \
    --add_generation_prompt
```

**Step 2: Training Average or Last Token model**
```bash
# Average representation
python scripts/cot_eval/train_multilayer_probe.py \
    --train_hidden_states_folder [HIDDEN_STATES_FOLDER] \
    --use_mean_representation

# Last Token representation
python scripts/cot_eval/train_multilayer_probe.py \
    --train_hidden_states_folder [HIDDEN_STATES_FOLDER] \
    --token_id 0
```

### 5. Math Datasets Preprocessing
Preprocessing scripts for math benchmarks:
```bash
python scripts/math_datasets_preprocessing/preprocess_process_bench.py
```

### 6. Plotting
Plotting scripts are available in two locations:
- `scripts/plots/`: Main scripts for generating comparative figures and result tables.
- `scripts/cot_eval/plot_harmfulness_over_tokens.py`: Specifically for visualizing the evolution of harmfulness scores over individual CoT trajectories.

Example:
```bash
python scripts/plots/plot_faithful_cot.py --tasks wildguardtest aegis_test
```

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{
    chrabaszcz2026monitoring,
    title={Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics},
    author={Maciej Chrabaszcz and Aleksander Szymczyk and Marcin Sendera and Tomasz Trzcinski and Sebastian Cygert},
    booktitle={Mechanistic Interpretability Workshop at ICML 2026},
    year={2026},
    url={https://openreview.net/forum?id=JpBsIqaDYI}
}
```