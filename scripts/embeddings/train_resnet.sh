python -m cnn.extract_resnet_embeddings \
    --dataset cifar10 \
    --mode train \
    --backbone resnet101 \
    --embedding-dim 15 \
    --epochs 20