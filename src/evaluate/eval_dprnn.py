import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

from src.models.dprnn import DPRNNSeparator
from src.data.librimix_dataloader import create_librimix_dataloader
from src.utils.logger import setup_logger
from src.utils.train_utils import set_training_device
from src.utils.eval_utils import pit_si_snr_loss


def build_model_from_config(model_config, num_sources):
    return DPRNNSeparator(
        num_sources=num_sources,
        enc_dim=model_config['enc_dim'],
        feature_dim=model_config['feature_dim'],
        hidden_dim=model_config['hidden_dim'],
        layers=model_config['layers'],
        segment_size=model_config['segment_size'],
        win_len=model_config['win_len'],
        rnn_type=model_config.get('rnn_type', 'LSTM'),
    )


def evaluate(model_dir: str, root_dir_data: str, config_data: str, split: str, device, logger):
    model_dir = Path(model_dir)
    # load training config and checkpoint
    config_json = model_dir / 'config.json'
    ckpt_path = model_dir / 'best_model.pth'
    if not config_json.exists() or not ckpt_path.exists():
        raise FileNotFoundError("Expected config.json and best_model.pth in model_dir")

    with open(config_json, 'r') as f:
        full_cfg = json.load(f)
    dataset_cfg = full_cfg['dataset']
    model_cfg = full_cfg['model']

    # dataloader
    dl = create_librimix_dataloader(root_dir_data=root_dir_data, config_path_data=config_data, split=split)
    logger.info(f"Loaded {len(dl.dataset)} {split} samples")

    # model
    model = build_model_from_config(model_cfg, num_sources=dataset_cfg['n_src'])
    model, device = set_training_device(logger, model, device='cpu' if device.type == 'mps' else device.type)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # loop
    all_scores = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"evaluating {split}"):
            mix = batch['mixture'].to(device)
            tgt = batch['sources'].to(device)
            est = model(mix)
            score = pit_si_snr_loss(est, tgt)  # [B]
            all_scores.append(score)

    all_scores = torch.cat(all_scores, dim=0)
    results = {
        'split': split,
        'num_samples': len(all_scores),
        'overall_mean_si_snr': all_scores.mean().item(),
        'overall_std_si_snr': all_scores.std().item(),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate DPRNN model on LibriMix')
    parser.add_argument('--model-dir', required=True, help='Directory with best_model.pth and config.json')
    parser.add_argument('--root-dir-data', required=True, help='Path to LibriMix root directory')
    parser.add_argument('--config-data', required=True, help='Path to dataset config YAML')
    parser.add_argument('--split', default='dev', choices=['dev', 'test'])
    parser.add_argument('--log-level', default='INFO')
    parser.add_argument('--log-file', default=None)

    args = parser.parse_args()
    logger = setup_logger(__name__, log_file=args.log_file, level=args.log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    logger.info(f"Device: {device}")

    results = evaluate(args.model_dir, args.root_dir_data, args.config_data, args.split, device, logger)
    out_file = Path(args.model_dir) / f"metrics_{args.split}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
