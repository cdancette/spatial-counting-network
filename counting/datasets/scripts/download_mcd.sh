#!/bin/bash
mkdir -p data/vqa/tallyqa/processed
cd data/vqa/tallyqa/processed
wget http://data.lip6.fr/dancette/tallyqa/tallyqa-ablated.zip
unzip tallyqa-ablated.zip
