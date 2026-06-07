#!/bin/bash

SRC_DIR="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data/color"
DST_DIR="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data/subset"

mkdir -p "$DST_DIR"

for ((i=0; i<100; i++)); do
    idx=$((i * 5))
    filename=$(printf "%05d.jpg" "$idx")
    cp "$SRC_DIR/$filename" "$DST_DIR/"
done