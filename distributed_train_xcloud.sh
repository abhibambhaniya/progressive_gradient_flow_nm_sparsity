#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC /workdir/pytorch-image-models-sparsity/train.py "$@"

