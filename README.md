# Midi Transformer

## How to run

### Local

First you need to create the environment with conda

`conda env create -f environment.yml`

data_config.json stores pre-processing information

train_config.json stores the training parameters

To process the data, run. The following commmand fill download all the files specified in data_config.json, and preprocesses them using REMI

`python data_pipeline.py --download --config data_config.json`

The tokenization is also ran through data_pipeline.py

`python data_pipeline.py --tokenize --config data.json`

To train a model, run

`python train.py`

### Triton

mamba env create -f environment.yml
source activate midi-transformer
python data_pipeline.py --download --config data_config.json
python data_pipeline.py --tokenize --config data_config.json
python train.py

download data

tokenize & train

TODO:

*preprocessing*
- add remi alternatives and argument parameter
- check config if it is confusing

*model*
- write transformer

*training*
- add config argument to initialization


