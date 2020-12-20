#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  echo "Missing one required argument"
  echo "------"
  echo "Usage: ./import_model.sh NAME MODEL"
  echo "NAME: Name/Path (without extension) of the model file you want to import"
  echo "MODEL: The model you want to train the classifier on. Defaults to LSTM"
  echo "------"
  exit 1
fi

if [ $# -eq 1 ]; then
  MODEL="lstm"

else
  MODEL=$2
fi


NAME=$1.tarjani
INTENT=intents

echo "Uncompressing the model files"
tar -zxvf $NAME
echo "Sorting the model files and cleaning up"
cp -r $INTENT/* ../intents/
rm -rf intents
python3 train_after_import.py --model $MODEL
