#!/bin/bash

DIR="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data/color"

cd "$DIR" || exit 1

for file in *.jpg; do
    num="${file%.jpg}"
    new_name=$(printf "%05d.jpg" "$num")
    mv "$file" "$new_name"
done