#!/bin/bash
python pretrain.py --augment_1 node_drop --augment_2 node_drop &
python pretrain.py --augment_1 node_drop --augment_2 node_mask &
python pretrain.py --augment_1 node_drop --augment_2 identity &
python pretrain.py --augment_1 node_drop --augment_2 subtree_mask &

python pretrain.py --augment_1 node_mask --augment_2 node_drop &
python pretrain.py --augment_1 node_mask --augment_2 node_mask &
python pretrain.py --augment_1 node_mask --augment_2 identity &
python pretrain.py --augment_1 node_mask --augment_2 subtree_mask &

python pretrain.py --augment_1 identity --augment_2 node_drop &
python pretrain.py --augment_1 identity --augment_2 node_mask &
python pretrain.py --augment_1 identity --augment_2 identity &
python pretrain.py --augment_1 identity --augment_2 subtree_mask &

python pretrain.py --augment_1 subtree_mask --augment_2 node_drop &
python pretrain.py --augment_1 subtree_mask --augment_2 node_mask &
python pretrain.py --augment_1 subtree_mask --augment_2 identity &
python pretrain.py --augment_1 subtree_mask --augment_2 subtree_mask &