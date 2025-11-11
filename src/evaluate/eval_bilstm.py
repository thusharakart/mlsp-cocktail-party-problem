import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from src.models.bilstm import BiLSTMSeparator
from src.utils.logger import setup_logger
from src.utils.eval_utils import pit_si_snr_loss
from src.utils.train_utils import count_parameters, set_device, set_seed, AverageMeter


def evaluate_bilstm(model, dataloader, device, logger=None):
    """
    Evaluate BiLSTM model on specified dataset split (matches training validation loop)

    Args:
        model: BiLSTMSeparator instance
        dataloader: dataloader for evaluation
        device: torch device
        logger: logger instance

    Returns:
        results: Dictionary with metrics
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    si_snr_meter = AverageMeter("val_si_snr")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="evaluating", leave=False, position=0)
        for batch in pbar:
            mixture = batch['mixture'].to(device)
            sources = batch['sources'].to(device)

            # forward pass
            separated = model(mixture)

            # use PIT loss (returns SI-SNR)
            si_snr = pit_si_snr_loss(separated, sources)
            loss = -si_snr.mean()  # negative because we minimize loss but maximize SI-SNR

            loss_meter.update(loss.item(), mixture.size(0))
            si_snr_meter.update(si_snr.mean().item(), mixture.size(0))

            pbar.set_postfix({
                "val_loss": f"{loss_meter.avg:.4f}",
                "si_snr": f"{si_snr_meter.avg:.4f}"
            })

    results = {
        "loss": loss_meter.avg,
        "si_snr": si_snr_meter.avg
    }

    if logger:
        logger.info(f"Evaluation results:")
        logger.info(f"  Loss: {results['loss']:.4f}")
        logger.info(f"  SI-SNR: {results['si_snr']:.4f} dB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM model on LibriMix")

    # required arguments
    parser.add_argument('--root-dir-data', required=True, help='Path to LibriMix root directory')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config-data', required=True, help='Path to data config YAML file')

    # optional arguments
    parser.add_argument('--split', default='dev', choices=['dev', 'test'], help='Dataset split to evaluate on (default: dev)')
    parser.add_argument('--output-dir', default='output/results/bilstm', help='Directory to save results')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of workers for dataloader (default: auto-detect, use 0 for reproducibility)')
    parser.add_argument('--log-level', default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--log-file', default=None, help='Optional log file path')

    args = parser.parse_args()

    # setup logger
    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)

    logger.info("="*50)
    logger.info("BiLSTM Evaluation")
    logger.info("="*50)
    logger.info(f"checkpoint: {args.checkpoint}")
    logger.info(f"root_dir_data: {args.root_dir_data}")
    logger.info(f"config_data: {args.config_data}")
    logger.info(f"split: {args.split}")

    # load checkpoint to get configs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    full_config = checkpoint.get('config', {})

    if not full_config:
        logger.error("Config not found in checkpoint. Cannot proceed.")
        raise ValueError("Config not found in checkpoint")

    model_config = full_config.get('model', {})
    dataset_config = full_config.get('dataset', {})
    run_config = full_config.get('run', {})

    if not model_config or not dataset_config:
        logger.error("Model or dataset config missing from checkpoint")
        raise ValueError("Model or dataset config missing from checkpoint")

    # set seed for reproducibility (use training seed from config)
    seed = run_config.get('seed', 42)
    set_seed(logger, seed)
    logger.info(f"Set random seed to {seed} for reproducible evaluation")

    # create model with exact same config as training
    logger.debug("-"*50)
    logger.debug("Creating BiLSTMSeparator model...")
    model = BiLSTMSeparator(
        num_sources=dataset_config.get('n_src', 2),
        num_layers=model_config.get('num_layers', 3),
        hidden_size=model_config.get('hidden_size', 512),
        dropout=model_config.get('dropout', 0.5),
        n_fft=model_config.get('n_fft', 1024),
        hop_length=model_config.get('hop_length', 256),
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    # load checkpoint state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint best SI-SNR: {checkpoint.get('best_si_snr', 'unknown'):.4f} dB")

    # move model to device (use training device from config)
    training_device = run_config.get('device', None)
    model, device = set_device(logger, model, device=training_device)

    if device.type == 'mps':
        logger.info("MPS HYBRID MODE ENABLED")
        logger.info("STFT and iSTFT computation: CPU")
        logger.info("Everything else in forward pass: MPS")

    # load dataloader using exact same config as training
    logger.debug("-"*50)
    logger.debug(f"Loading {args.split} dataloader...")

    # load config to potentially override num_workers
    import yaml
    with open(args.config_data) as f:
        eval_config = yaml.safe_load(f)

    # override num_workers if specified
    if args.num_workers is not None:
        eval_config['dataloader']['num_workers'] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    # create dataset and dataloader manually for more control
    from torch.utils.data import DataLoader
    from src.data.librimix_dataset import LibriMixDataset, collate_fn_librimix

    dataset = LibriMixDataset(
        root_dir_data=args.root_dir_data,
        config_path_data=args.config_data,
        split=args.split,
    )

    batch_sizes = {'dev': eval_config['dataloader']['batch_size_val'],
                   'test': eval_config['dataloader']['batch_size_test']}
    batch_size = batch_sizes[args.split]

    num_workers = args.num_workers if args.num_workers is not None else eval_config['dataloader'].get('num_workers', None)
    if num_workers is None:
        import os
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, 8)

    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    pin_memory = has_gpu and num_workers > 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_librimix,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(f"Loaded {len(dataloader.dataset)} samples from {args.split} split")
    logger.info(f"Using {num_workers} workers (warning: may cause SI-SNR variation due to random crop sampling)")

    # run evaluation (same structure as training validation loop)
    logger.info("="*50)
    logger.info(f"Evaluating on {args.split} split...")
    results = evaluate_bilstm(
        model=model,
        dataloader=dataloader,
        device=device,
        logger=logger
    )

    # add metadata to results
    results['checkpoint'] = str(args.checkpoint)
    results['split'] = args.split
    results['model_config'] = model_config
    results['dataset_config'] = dataset_config
    results['run_config'] = run_config

    # save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_file = metrics_dir / f'bilstm_results_{args.split}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
