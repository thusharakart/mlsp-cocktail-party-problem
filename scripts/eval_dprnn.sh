CUDA_VISIBLE_DEVICES=0 python -m src.evaluate.eval_dprnn \
  --model-dir output/models/dprnn/dprnn_20251107_2175f5 \
  --root-dir-data data/Libri2Mix \
  --config-data config/libri2mix_16k_2src.yaml \
  --split test