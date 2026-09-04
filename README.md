## Hardware and execution time

All experiments were conducted on the following local hardware:
* **CPU:** Intel Core i7-12700H
# Noise Evaluation for Anomaly Detection

A small Python/PyTorch project that trains a neural network to recognize feature-level noise and uses its predictions as anomaly scores. It follows the noise-evaluation approach from a research paper and compares the results with Isolation Forest and Local Outlier Factor.

## What it does

The experiment pipeline is:

1. Load an ADBench dataset from `data/`.
2. Prepare the data using either one-class classification (OCC) or unsupervised anomaly detection (UAD).
3. Generate noisy versions of training samples with configurable noise distributions and noise ratios.
4. Train a Vanilla MLP or residual MLP with PyTorch.
5. Evaluate ROC-AUC scores for the neural model and two classical baselines.
6. Save the configuration and results under `runs/`.

## Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- Pydantic and YAML configuration

## Setup

```bash
pip install -r requirements.txt
```

Download an ADBench Classical dataset, such as `6_cardio.npz`, and place it in `data/`:

```text
data/6_cardio.npz
```

Datasets supported by the current configuration include `6_cardio`, `30_satellite`, and `26_optdigits`.

## Run an experiment

Edit `config.yaml` to select the dataset, training type, model, and noise settings. Then run:

```bash
python main.py --config config.yaml
```

The default configuration runs UAD with a Vanilla MLP and uniform noise. Set `training.training_type` to `OCC` to train only on normal samples, or set `training.model_type` to `RES_MLP` to use the residual network.

When `experiment.write_to_file` is enabled, each run creates a timestamped folder containing the configuration and results.

## Example results

Representative ROC-AUC scores from the included experiments:

| Dataset | Protocol | Model | This project | Isolation Forest | LOF |
| --- | --- | --- | ---: | ---: | ---: |
| 6_cardio | OCC | ResMLP | **0.9582** | 0.9565 | 0.9363 |
| 6_cardio | UAD | VanillaMLP | 0.9116 | **0.9281** | 0.6305 |
| 26_optdigits | OCC | VanillaMLP | 0.9065 | 0.8417 | **0.9663** |

The experiments also show an important practical difference: OCC performs more consistently than UAD because UAD training data can contain anomalies that the model treats as normal.

## Project structure

```text
main.py       Run the complete experiment
config.py     Validate and load configuration
data.py       Load and prepare ADBench data
noise.py      Generate configurable feature noise
model.py      Vanilla and residual MLP models
train.py      Training loop
evaluate.py   AUC evaluation and baselines
```
The architectural decisions (config structure, class-based modules, noise generator API, OCC vs UAD interpretation), experimental design (which datasets, which configurations to compare), and the critique/observations in this README are mine. The code itself was written by Claude based on my direction. I did not find or use any public implementation of the paper.
