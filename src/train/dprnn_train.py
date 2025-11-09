import argparse
import json
import yaml
from pathlib import Path
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from time import time

from src.models.dprnn import DPRNNSeparator
from src.utils.logger import setup_logger
from src.data.librimix_dataloader import create_train_val_test_loaders
from src.utils.train_utils import (
    set_training_device,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    AverageMeter,
    count_parameters,
    save_training_config,
    generate_unique_model_id,
)
from src.utils.eval_utils import pit_si_snr_loss
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


def train_epoch(logger, model, dataloader, optimizer, device, gradient_clip_norm=5.0, max_batches=None, writer=None, global_step=0, use_amp=False, scaler=None, grad_accum_steps=1):
    model.train()
    loss_meter = AverageMeter("train_loss")
    grad_norm_meter = AverageMeter("grad_norm")

    pbar = tqdm(dataloader, desc="training", leave=False, position=0)
    optimizer.zero_grad()
    last_logged_grad_norm = 0.0
    for batch_idx, batch in enumerate(pbar):
        if max_batches and batch_idx >= int(max_batches):
            break

        mixture = batch['mixture'].to(device)
        sources = batch['sources'].to(device)

        step_boundary = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == len(dataloader) - 1)

        if use_amp and scaler is not None and autocast is not None:
            with autocast():
                separated = model(mixture)
                si_snr = pit_si_snr_loss(separated, sources)
                loss = -si_snr.mean()
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if step_boundary:
                if gradient_clip_norm and gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                else:
                    grad_norm = torch.tensor(0.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                last_logged_grad_norm = float(getattr(grad_norm, 'item', lambda: grad_norm)()) if hasattr(grad_norm, 'item') else float(grad_norm)
        else:
            separated = model(mixture)
            si_snr = pit_si_snr_loss(separated, sources)
            loss = -si_snr.mean()
            (loss / grad_accum_steps).backward()
            if step_boundary:
                if gradient_clip_norm and gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()
                optimizer.zero_grad()
                last_logged_grad_norm = float(getattr(grad_norm, 'item', lambda: grad_norm)()) if hasattr(grad_norm, 'item') else float(grad_norm)

        if writer and batch_idx % 50 == 0:
            writer.add_scalar('loss/train_batch', loss.item(), global_step)
            writer.add_scalar('gradient/norm_batch', last_logged_grad_norm, global_step)

        loss_meter.update(loss.item(), mixture.size(0))
        grad_norm_meter.update(last_logged_grad_norm, 1)
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
        global_step += 1

    return {"loss": loss_meter.avg, "avg_grad_norm": grad_norm_meter.avg}, global_step


def validate(model, dataloader, device, use_amp=False):
    model.eval()
    loss_meter = AverageMeter("val_loss")
    si_snr_meter = AverageMeter("val_si_snr")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="validating", leave=False, position=0)
        for batch in pbar:
            mixture = batch['mixture'].to(device)
            sources = batch['sources'].to(device)
            if use_amp and autocast is not None:
                with autocast():
                    separated = model(mixture)
                    si_snr = pit_si_snr_loss(separated, sources)
            else:
                separated = model(mixture)
                si_snr = pit_si_snr_loss(separated, sources)
            loss = -si_snr.mean()

            loss_meter.update(loss.item(), mixture.size(0))
            si_snr_meter.update(si_snr.mean().item(), mixture.size(0))
            pbar.set_postfix({"val_loss": f"{loss_meter.avg:.4f}", "si_snr": f"{si_snr_meter.avg:.4f}"})

    return {"loss": loss_meter.avg, "si_snr": si_snr_meter.avg}


def main():
    parser = argparse.ArgumentParser(description="Train DPRNN (FaSNet-style) separator on LibriMix")

    # required arguments
    parser.add_argument("--root-dir-data", required=True, help="Path to LibriMix root directory")
    parser.add_argument("--config-data", required=True, help="Path to dataset config YAML")
    parser.add_argument("--config-model", required=True, help="Path to DPRNN model config YAML")

    # checkpointing
    parser.add_argument("--save-dir", default="output/models/dprnn", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--save-checkpoints", action="store_true", help="Enable periodic checkpoint saving")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    # other
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--log-file', default=None, help='Optional log file path')
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--max-train-batches", default=None, help="Max batches per epoch (for fast debugging)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Accumulate gradients over N steps before optimizer step")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (CUDA only)")

    args = parser.parse_args()

    # logger
    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)
    logger.info("=" * 50)
    logger.info("DPRNN (FaSNet) Training")
    logger.info("=" * 50)

    # load configs
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt_path = Path(args.resume)
        model_dir = str(ckpt_path.parent)
        config_path = ckpt_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}. Cannot resume without original config.")
        with open(config_path, 'r') as f:
            full_training_config = json.load(f)
        dataset_config = full_training_config['dataset']
        run_config = full_training_config['run']
        model_config = full_training_config['model']
        training_config = full_training_config['training']
        run_id = full_training_config.get('run', {}).get('run_id', 'resumed_run')
        logger.info(f"Loaded config from checkpoint. Resuming run: {run_id}")
    else:
        with open(args.config_data) as f:
            full_data_config = yaml.safe_load(f)
        dataset_config = full_data_config['dataset']

        with open(args.config_model) as f:
            full_model_config = yaml.safe_load(f)

        run_config = full_model_config['run']
        model_config = full_model_config['model']
        training_config = full_model_config['training']

        model_type = Path(args.save_dir).name
        model_dir, run_id = generate_unique_model_id(logger, model_type, args.save_dir)
        full_training_config = full_data_config | full_model_config
        full_training_config['run']['run_id'] = run_id
        save_training_config(logger, full_training_config, model_dir)

    set_seed(logger, run_config['seed'])

    # tensorboard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = Path(model_dir) / "tensorboard"
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs: {tensorboard_dir}")

    # dataloaders
    train_loader, val_loader = create_train_val_test_loaders(root_dir_data=args.root_dir_data, config_path_data=args.config_data, include_test=False)
    logger.debug(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # model
    logger.debug("Creating DPRNNSeparator model...")
    model = DPRNNSeparator(
        num_sources=dataset_config['n_src'],
        enc_dim=model_config['enc_dim'],
        feature_dim=model_config['feature_dim'],
        hidden_dim=model_config['hidden_dim'],
        layers=model_config['layers'],
        segment_size=model_config['segment_size'],
        win_len=model_config['win_len'],
        rnn_type=model_config.get('rnn_type', 'LSTM'),
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    model, device = set_training_device(logger, model, device=run_config['device'])

    # optimizer & scheduler
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP setup
    use_amp = bool(args.amp and torch.cuda.is_available() and GradScaler is not None)
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("AMP enabled: mixed precision training active")
    else:
        logger.info("AMP disabled (use --amp on CUDA to enable)")

    epochs = int(training_config['epochs'])
    scheduler_name = training_config['scheduler']
    scheduler_params = training_config['scheduler_params'][scheduler_name]
    if scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_size=scheduler_params['step_size'], gamma=scheduler_params['gamma'])
    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_params['factor'], patience=scheduler_params['patience'], verbose=True)
    else:
        scheduler = None

    # resume
    start_epoch = 1
    best_si_snr = float('-inf')
    if args.resume:
        _, start_epoch, best_si_snr = load_checkpoint(logger, args.resume, model, optimizer, device)
        logger.info(f"Resuming from epoch {start_epoch}, best SI-SNR: {best_si_snr:.4f} dB")

    # train loop
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)

    patience_counter = 0
    global_step = 0
    for epoch in range(start_epoch, epochs + 1):
        logger.info("")
        logger.info(f"Epoch {epoch}/{epochs}")

        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar('learning_rate', current_lr, epoch)

        t1 = time()
        train_metrics, global_step = train_epoch(
            logger,
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip_norm=training_config['gradient_clip_norm'],
            max_batches=args.max_train_batches,
            writer=writer,
            global_step=global_step,
            use_amp=use_amp,
            scaler=scaler,
            grad_accum_steps=max(1, int(args.grad_accum_steps)),
        )
        logger.info(f"Train loss: {train_metrics['loss']:.4f}, time: {time() - t1:.2f}s")
        if writer:
            writer.add_scalar('loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('gradient/norm_avg', train_metrics['avg_grad_norm'], epoch)

        t2 = time()
        val_metrics = validate(model, val_loader, device, use_amp=use_amp)
        logger.info(f"Val loss: {val_metrics['loss']:.4f}, SI-SNR: {val_metrics['si_snr']:.4f} dB, time: {time() - t2:.2f}s")
        if writer:
            writer.add_scalar('loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('si_snr/val', val_metrics['si_snr'], epoch)

        # step scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # best check
        is_best = val_metrics['si_snr'] > best_si_snr
        if is_best:
            best_si_snr = val_metrics['si_snr']
            patience_counter = 0
            logger.info(f"New best model (SI-SNR: {best_si_snr:.4f} dB)")
        else:
            patience_counter += 1

        # save
        should_save_periodic = args.save_checkpoints and (epoch % args.save_every == 0)
        if should_save_periodic or is_best:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_si_snr': best_si_snr,
                'config': (full_training_config if not args.resume else None),
            }
            if should_save_periodic:
                save_checkpoint(logger, state, model_dir, filename=f"checkpoint_epoch_{epoch}.pth", is_best=False)
            if is_best:
                save_checkpoint(logger, state, model_dir, filename="best_model.pth", is_best=True)

        if patience_counter >= training_config['early_stopping_patience']:
            logger.info(f"Early stopping triggered (patience={training_config['early_stopping_patience']})")
            break

    if writer:
        writer.close()

    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info(f"Best SI-SNR: {best_si_snr:.4f} dB")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
