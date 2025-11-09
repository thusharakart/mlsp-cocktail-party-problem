CUDA_VISIBLE_DEVICES=1 python -m src.train.dprnn_train \
  --root-dir-data data/Libri2Mix \
  --config-data config/libri2mix_16k_2src.yaml \
  --config-model config/dprnn.yaml \
  --save-dir output/models/dprnn \
  --tensorboard 
