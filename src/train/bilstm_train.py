import argparse
import json
import yaml
import csv
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from time import time

from src.models.bilstm import BiLSTMSeparator
from src.utils.logger import setup_logger
from src.data.librimix_dataloader import create_train_val_test_loaders
from src.utils.train_utils import (
    set_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    AverageMeter,
    count_parameters,
    save_training_config,
    generate_unique_model_id
)
from src.utils.eval_utils import pit_si_snr_loss


def train_epoch(logger, model, dataloader, optimizer, device, gradient_clip_norm=5.0, max_batches=None, writer=None, global_step=0):
    """
    Train for one epoch

    Args:
        model: BiLSTMSeparator instance
        dataloader: training dataloader
        optimizer: optimizer instance
        device: torch device
        gradient_clip_norm: (default: 5.0)
        max_batches: If set, only train on this many batches per epoch
    Returns:
        dict with training metrics
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    grad_norm_meter = AverageMeter("grad_norm")

    pbar = tqdm(dataloader, desc="training", leave=False, position=0)
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break
        
        mixture = batch['mixture'].to(device)
        sources = batch['sources'].to(device)

        optimizer.zero_grad()

        # forward pass
        separated = model(mixture)
        
        # use PIT loss (returns SI-SNR)
        si_snr = pit_si_snr_loss(separated, sources)
        loss = -si_snr.mean() # negative because we minimize loss but maximize SI-SNR

        # backward pass
        loss.backward()
        
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        # log to TensorBoard every N batches
        if writer and batch_idx % 50 == 0:
            writer.add_scalar('loss/train_batch', loss.item(), global_step)
            writer.add_scalar('gradient/norm_batch', grad_norm.item(), global_step)

        optimizer.step()
        if grad_norm > gradient_clip_norm * 2:
            logger.warning(f"Large gradient: {grad_norm:.2f}")

        loss_meter.update(loss.item(), mixture.size(0))
        grad_norm_meter.update(grad_norm.item(), 1) 
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        global_step += 1

    return {
        "loss": loss_meter.avg,
        "avg_grad_norm": grad_norm_meter.avg
    }, global_step


def validate(model, dataloader, device):
    """
    Validate on one epoch

    Args:
        model: BiLSTMSeparator instance
        dataloader: validation dataloader
        device: torch device
    Returns:
        dict with validation metrics
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    si_snr_meter = AverageMeter("val_si_snr")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="validating", leave=False, position=0)
        for batch in pbar:
            mixture = batch['mixture'].to(device)
            sources = batch['sources'].to(device)

            # forward pass
            separated = model(mixture)
            
            # use PIT loss
            si_snr = pit_si_snr_loss(separated, sources)
            loss = -si_snr.mean()

            loss_meter.update(loss.item(), mixture.size(0))
            si_snr_meter.update(si_snr.mean().item(), mixture.size(0))

            pbar.set_postfix({
                "val_loss": f"{loss_meter.avg:.4f}", 
                "si_snr": f"{si_snr_meter.avg:.4f}"
            })

    return {"loss": loss_meter.avg, "si_snr": si_snr_meter.avg}


def main():
    ### PARSE ARGS
    parser = argparse.ArgumentParser(description="Train BiLSTM source separator on LibriMix")

    # required arguments
    parser.add_argument("--root-dir-data", required=True, help="Path to LibriMix root directory")
    parser.add_argument("--config-data", required=True, help="Path to dataset config YAML")
    parser.add_argument("--config-model", required=True, help="Path to BiLSTM model config YAML")

    # checkpointing
    parser.add_argument("--save-dir", default="output/models/bilstm", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--save-checkpoints", action="store_true", help="Enable periodic checkpoint saving")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    # other
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help=(
            "Set the logging verbosity level (default: INFO). "
            "Use DEBUG for detailed debugging information, "
            "INFO for general progress messages, "
            "WARNING for potential issues, "
            "ERROR for serious problems, "
            "and CRITICAL for fatal errors only"
        )
    )
    parser.add_argument('--log-file', default=None, help='Optional log file path')
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--max-train-batches", default=None, help="Max batches per epoch (for fast debugging)")

    args = parser.parse_args()


    ### SETUP
    # setup logger
    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)
    logger.info("="*50)
    logger.info("BiLSTM Source Separator Training")
    logger.info("="*50)
    
    #**DETERMINE IF RESUMING AND LOAD APPROPRIATE CONFIGS**
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        
        # extract model directory from checkpoint path
        checkpoint_path = Path(args.resume)
        model_dir = str(checkpoint_path.parent)
        
        # load config from checkpoint directory
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Cannot resume without original config."
            )
        
        with open(config_path, 'r') as f:
            full_training_config = json.load(f)
        
        # extract sub-configs
        dataset_config = full_training_config['dataset']
        run_config = full_training_config['run']
        model_config = full_training_config['model']
        training_config = full_training_config['training']
        
        run_id = full_training_config.get('run', {}).get('run_id', 'resumed_run')

        logger.info(f"Loaded config from checkpoint")
        logger.info(f"Resuming run: {run_id}")
        logger.info(f"Using existing directory: {model_dir}")
        
    else:
        logger.info("Starting fresh training")
        
        # koad configs from YAML files
        with open(args.config_data) as f:
            full_data_config = yaml.safe_load(f)
        dataset_config = full_data_config['dataset']

        with open(args.config_model) as f:
            full_model_config = yaml.safe_load(f)

        run_config = full_model_config['run']
        model_config = full_model_config['model']
        training_config = full_model_config['training']
        
        # generate unique model ID
        model_type = Path(args.save_dir).name
        model_dir, run_id = generate_unique_model_id(logger, model_type, args.save_dir)
        
        # merge configs and save
        full_training_config = full_data_config | full_model_config
        full_training_config['run']['run_id'] = run_id
        save_training_config(logger, full_training_config, model_dir)
    
    logger.debug("Model config:")
    for key, value in model_config.items():
        logger.debug(f"  {key}: {value}")
    logger.debug("Training config:")
    for key, value in training_config.items():
        logger.debug(f"  {key}: {value}")
    
    # set seed
    set_seed(logger, run_config['seed'])

    # setup tensorboard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = Path(model_dir) / "tensorboard"
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
        logger.info(f"View with: tensorboard --logdir {tensorboard_dir}")
    else:
        logger.debug("TensorBoard logging disabled (use --tensorboard to enable)")


    ### LOAD DATALOADERS
    logger.debug("-"*50)
    logger.debug(f"Loading dataloaders from {args.root_dir_data}...")
    train_loader, val_loader = create_train_val_test_loaders(root_dir_data=args.root_dir_data, config_path_data=args.config_data, include_test=False)

    logger.debug(f"Train samples: {len(train_loader.dataset)}")
    logger.debug(f"Val samples: {len(val_loader.dataset)}")


    ### CREATE MODEL
    logger.debug("-"*50)
    logger.debug("Creating BiLSTMSeparator model...")
    model = BiLSTMSeparator(
        num_sources=dataset_config['n_src'],
        num_layers=model_config['num_layers'],
        hidden_size=model_config['hidden_size'],
        dropout=model_config['dropout'],
        n_fft=model_config['n_fft'],
        hop_length=model_config['hop_length'],
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    model, device = set_device(logger, model, device=run_config['device'])
    if device.type == 'mps':
        logger.info("MPS HYBRID MODE ENABLED")
        logger.info("STFT and ISFT computation: CPU")
        logger.info("Everything else in forward pass: MPS")
        logger.info("This is faster than pure CPU while working around MPS limitations")

    # setup optimizer
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.debug(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")

    # setup scheduler
    epochs = int(training_config['epochs'])
    scheduler = training_config['scheduler']
    scheduler_config = training_config['scheduler_params'][scheduler]
    if scheduler == "step":
        step_size = scheduler_config['step_size']
        gamma = scheduler_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.debug(f"Scheduler: StepLR ({step_size=}, {gamma=})")
    elif scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        logger.debug(f"Scheduler: CosineAnnealingLR (T_max={epochs})")
    elif scheduler == "reduce_on_plateau":
        factor = scheduler_config['factor']
        patience = scheduler_config['patience']
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
        logger.debug(f"Scheduler: ReduceLROnPlateau ({patience=})")
    else:
        scheduler = None
        logger.debug("Scheduler: None")


    ### IF LOADING CHECKPOINTED, RESUME FROM CHECKPOINT
    start_epoch = 1
    best_si_snr = float("-inf")
    if args.resume:
        checkpoint, start_epoch, best_si_snr = load_checkpoint(
            logger, args.resume, model, optimizer, device
        )
        logger.info(f"Resuming from epoch {start_epoch}, best SI-SNR: {best_si_snr:.4f} dB")

    ### SETUP TRAINING LOGS
    logs_dir = Path(model_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_file = logs_dir / "training_metrics.csv"

    # create CSV header if file doesn't exist
    metrics_written = metrics_log_file.exists()

    ### TRAINING LOOP
    logger.info("="*50)
    logger.info("Starting training...")
    logger.info("="*50)

    if args.save_checkpoints:
        logger.info(f"Checkpointing: saving every {args.save_every} epochs + best model")
    else:
        logger.info("Checkpointing: only saving best model")

    logger.info(f"Training metrics log: {metrics_log_file}")

    patience_counter = 0
    global_step = 0  # track total batches across epochs
    for epoch in range(start_epoch, epochs + 1):
        logger.info("")
        logger.info(f"Epoch {epoch}/{epochs}")

        # log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.debug(f"Learning rate: {current_lr:.6f}")
        if writer:
            writer.add_scalar('learning_rate', current_lr, epoch)

        # train epoch
        t1_train = time()
        train_metrics, global_step = train_epoch(logger, model, train_loader, optimizer, device, gradient_clip_norm=training_config['gradient_clip_norm'], max_batches=args.max_train_batches, writer=writer, global_step=global_step)
        tdiff_train = time() - t1_train
        logger.info(f"Train loss: {train_metrics['loss']:.4f}, Time: {tdiff_train:.2f}s")
        if writer:
            writer.add_scalar('loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('gradient/norm_avg', train_metrics['avg_grad_norm'], epoch)
            writer.add_scalar('time/train', tdiff_train, epoch)

        # validate
        t1_val = time()
        val_metrics = validate(model, val_loader, device)
        tdiff_val = time() - t1_val
        logger.info(f"Val loss: {val_metrics['loss']:.4f}, SI-SNR: {val_metrics['si_snr']:.4f} dB, Time: {tdiff_val:.2f}s")
        if writer:
            writer.add_scalar('loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('si_snr/val', val_metrics['si_snr'], epoch)
            writer.add_scalar('time/val', tdiff_val, epoch)
            # log comparison on same plot
            writer.add_scalars('loss/train_vs_val', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)

        # check if best model
        is_best = val_metrics["si_snr"] > best_si_snr
        if is_best:
            best_si_snr = val_metrics["si_snr"]
            patience_counter = 0
            logger.info(f"New best model (SI-SNR: {best_si_snr:.4f} dB)")
        else:
            patience_counter += 1

        # log metrics to CSV
        with open(metrics_log_file, 'a', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=[
                'epoch', 'train_loss', 'val_loss', 'val_si_snr', 'best_si_snr',
                'is_best', 'learning_rate', 'time_train', 'time_val'
            ])

            # write header on first epoch
            if epoch == start_epoch:
                writer_csv.writeheader()

            writer_csv.writerow({
                'epoch': epoch,
                'train_loss': f"{train_metrics['loss']:.6f}",
                'val_loss': f"{val_metrics['loss']:.6f}",
                'val_si_snr': f"{val_metrics['si_snr']:.6f}",
                'best_si_snr': f"{best_si_snr:.6f}",
                'is_best': is_best,
                'learning_rate': f"{current_lr:.6e}",
                'time_train': f"{tdiff_train:.2f}",
                'time_val': f"{tdiff_val:.2f}",
            })

        # update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # save checkpoint (only if enabled OR if best model)
        should_save_periodic = args.save_checkpoints and (epoch % args.save_every == 0)
        should_save_best = is_best

        if should_save_periodic or should_save_best:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_si_snr": best_si_snr,
                "config": full_training_config,
            }
            
            # save periodic checkpoint
            if should_save_periodic:
                checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
                save_checkpoint(logger, state, model_dir, filename=checkpoint_filename, is_best=False)
                logger.info(f"Saved checkpoint: {checkpoint_filename}")
            
            # save best model (separate file)
            if should_save_best:
                save_checkpoint(logger, state, model_dir, filename="best_model.pth", is_best=True)
                logger.info("Saved best model")

        # early stopping
        if patience_counter >= training_config['early_stopping_patience']:
            logger.info(f"Early stopping triggered (patience={training_config['early_stopping_patience']})")
            break
    
    if writer:
        writer.close()
        logger.debug("TensorBoard logs saved")

    logger.info("="*50)
    logger.info("Training complete!")
    logger.info(f"Best SI-SNR: {best_si_snr:.4f} dB")
    logger.info("="*50)


if __name__ == "__main__":
    main()
