**Midi Transformer**

First create env

mamba env create -f environment.yml
source activate midi-transformer
python data_pipeline.py --download --config data_config.json
python data_pipeline.py --tokenize --config data_config.json
python train.py

download data

tokenize & train




