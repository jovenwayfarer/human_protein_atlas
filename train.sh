#!/bin/bash

cell_tiles_dir = "$1"
for i in {0..4}; do \
    python main.py --fold $i --model efficientnet_b1 --data_dir "$cell_tiles_dir"; \
done

for i in {0..4}; do \
    python main.py --fold $i --model efficientnet_b0 --data_dir "$cell_tiles_dir"; \
done
