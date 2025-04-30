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


## Full process to launch a training:

### Create a tokenizer
1) Download the datasets

`python data_pipeline.py --download --config data_config.json`

- you can change the downloaded packages in the "data_config.json" file

2) Chunk data and train tokenizer

`python data_pipeline.py --chunk --config data_config.json`
- this will train the tokenizer with the **vocabulary size** set in "data_config.json"
- the tokenizer and the chunks will be saved in "training_outputs/tokenizers/"

### Launch a training
1) In the "train_config.json" you must specify the tokenizer that you use. Change the "tokenizer_dir" value to the name of the folder that contains the chunks and the tokenizer

*Example : 20250430_125539_tokenizer*

2) In the "train_config.json", you can change the parameters that you want to dimension the model

3) Running the actual training

`python train.py`

4) When everything is finished, you should have in the "training_outputs/model_data/" a new folder that contains a bunch of infos about the model:
    - All models of all epochs
    - The best model
    - the data_config.json
    - the train_config.json
    - the tokenizer used

All of these infos are useful for the generation that comes next.

### Make a generation

1) First, you must specify which model you will use, so in the "generation_config.json" change the value of "MODEL_DIR" to the name of the folder that contains all the informations

*Example : 20250430_134522_model_data*

2) Create the folders for generation

`python generate.py --create_folders`

- Afterward, you can place your input midi file in the folder named "generation/inputs"

3) Extend your midi file

`python generate.py --inname {name of your midi file} --outname {name of the result midi file} --nevent {number of event you want to add}`

- By default "--inname" and "--outname" are respectively "input" and "output" so the quick way is to name your input file "input.midi" and just run `python generate.py --nevent {number}`

- Your result midi file is available on the folder "generation/outputs"

**BE CAREFUL : if there is already a midi file with the name of the output in the folder it will be replaced**






TODO:

*preprocessing*
- add remi alternatives and argument parameter
- check config if it is confusing

*model*
- write transformer

*training*
- add config argument to initialization




