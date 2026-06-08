import argparse
import random
from datetime import datetime
import time

import numpy as np
import torch

from config import load_config
from data import ADBenchDataLoader
from train import AnomalyDetectorTrainer
from evaluate import AnomalyEvaluator
from logger import DualLogger
from enums import TrainingType


def set_seed(seed: int) -> None:
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Noise Evaluation Anomaly Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up loggers — shared run_id so config + results land in the same folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_type = cfg.training.training_type.value
    model_type = cfg.training.model_type.value

    config_logger, results_logger = create_loggers(cfg, run_id, training_type, model_type)

    # Print full config (console + config file)
    cfg.write_summary(config_logger.write)

    results_logger.write(f"--- Running experiment: {cfg.experiment.dataset_name} on {device} ---")
    results_logger.write(f"Training mode: {cfg.training.training_type.value}")

    t_total_start = time.time()

    # 1. Load and prepare data
    loader = ADBenchDataLoader(
        dataset_name=cfg.experiment.dataset_name,
        seed=cfg.experiment.seed,
    )

    if cfg.training.training_type == TrainingType.OCC:
        X_train, X_test, y_test = loader.prepare_occ()
    else:  # UAD
        X_train, X_test, y_test = loader.prepare_uad()

    results_logger.write(f"X_train shape: {tuple(X_train.shape)} | X_test shape: {tuple(X_test.shape)}")
    results_logger.write(f"Test anomaly ratio: {y_test.mean():.3f}")

    # 2. Train
    results_logger.write("\nTraining model...")
    trainer = AnomalyDetectorTrainer(
        training_config=cfg.training,
        noise_diversity_config=cfg.noise_diversity,
        input_dim=X_train.shape[1],
        device=device,
        logger=results_logger,
    )
    model = trainer.fit(X_train)

    # 3. Evaluate
    evaluator = AnomalyEvaluator(
        model=model,
        agg_type=cfg.experiment.agg_type,
        device=device,
        logger=results_logger,
    )
    results = evaluator.run(
        dataset_name=cfg.experiment.dataset_name,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )

    total_time = time.time() - t_total_start

    # 4. Final summary
    results_logger.write("\n=== Final Results ===")
    results_logger.write(f"Dataset: {results['dataset']}")
    results_logger.write(f"  Ours (NoiseEval) : AUC = {results['Ours']:.4f}  ({results['time_ours_sec']:.1f}s)")
    results_logger.write(f"  IsolationForest  : AUC = {results['IsolationForest']:.4f}  ({results['time_iforest_sec']:.1f}s)")
    results_logger.write(f"  LOF              : AUC = {results['LOF']:.4f}  ({results['time_lof_sec']:.1f}s)")

    results_logger.write(f"\nTotal experiment time: {total_time:.2f}s")
    
    if cfg.experiment.write_to_file:
        print(f"\nConfig written to: {config_logger.filepath}")
        print(f"Results written to: {results_logger.filepath}")

def create_loggers(cfg, run_id, training_type, model_type):
    config_logger = DualLogger(
        enabled=cfg.experiment.write_to_file,
        output_dir=cfg.experiment.output_dir,
        dataset_name=cfg.experiment.dataset_name,
        training_type=training_type,
        model_type=model_type,
        prefix="config",
        run_id=run_id,
    )
    results_logger = DualLogger(
        enabled=cfg.experiment.write_to_file,
        output_dir=cfg.experiment.output_dir,
        dataset_name=cfg.experiment.dataset_name,
        training_type=training_type,
        model_type=model_type,
        prefix="results",
        run_id=run_id,
    )
    
    return config_logger,results_logger


if __name__ == "__main__":
    main()