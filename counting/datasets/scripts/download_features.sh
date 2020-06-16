#!/bin/bash

mkdir -p data/vqa/coco/extract_rcnn
cd data/vqa/coco/extract_rcnn
wget http://data.lip6.fr/cadene/block/coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36.tar
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar


mkdir -p data/vqa/vgenome/extract_rcnn
cd data/vqa/vgenome/extract_rcnn
wget https://data.lip6.fr/cadene/block/vgenome/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36-small.zip
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar
