#!/bin/bash

set -e

if [ $# -eq 0 ]; then
  echo "Missing one required argument"
  echo "------"
  echo "Usage: ./export_model.sh NAME"
  echo "NAME: Name to give to your exported model"
  echo "------"
  exit 1
fi

NAME="$1"
EXTENSION=".tar.gz"
MODEL=$NAME$EXTENSION
echo "Exporting the model"
tar -czf $MODEL ../intents
mv $MODEL $NAME.tarjani
echo "Model export successful"
