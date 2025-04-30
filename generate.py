import argparse
import json
from pathlib import Path
from transformers import TransfoXLLMHeadModel
from data_pipeline import Config, TokenizerManager
import torch
import os
import time


def open_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

class Generator:
    def __init__(self):
        self.data_config = Config.load_from_file(Path("./data_config.json"))
        self.generation_config = open_json("./generation_config.json")
        self.token_manager = TokenizerManager(config=self.data_config)
        self.tokenizer = self.token_manager.load_tokenizer(Path(self.generation_config["MODEL_DIR"] + "/trained_tokenizer.json"))


        self.attention_mask = None
        self.input_ids = None
        self.output_ids = None

        self.model = TransfoXLLMHeadModel.from_pretrained(self.generation_config["MODEL_DIR"] + "/models/best_model")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    def extend(self,input_name, output_name, number_of_new_events):
        start_time = time.time()
        ## Getting the tokenized input
        self.input_ids = self.tokenizer.encode(self.generation_config["INPUTS_DIR"]+"/"+input_name +".mid", encode_ids=True)[0].ids
        self.input_ids = torch.tensor(self.input_ids).unsqueeze(0)

        ## Computing the predictions
        self.output_ids = self.model.generate(
                                    self.input_ids,
                                    max_new_tokens=number_of_new_events,
                                    do_sample=True,
                                    temperature=5.0,
                                    repetition_penalty=1.2,
                                    eos_token_id=self.tokenizer["EOS_None"],
                                    pad_token_id=self.tokenizer.pad_token_id
                                    )
            
        ## Getting a midi file back
        midi =self.tokenizer.decode(self.output_ids)

        # Saving the new midi file
        midi.dump_midi(self.generation_config["OUTPUTS_DIR"]+"/"+output_name +".mid")

        end_time = time.time()
        print(end_time-start_time)


def main():
    argparser = argparse.ArgumentParser(description="Module for generation of a bigger MIDI file")

    
    argparser.add_argument('--inname', type=str, default="input")
    argparser.add_argument('--outname', type=str, default= "output")
    argparser.add_argument('--nevents', type=int, default=1)

    argparser.add_argument('--create_folders', action='store_true')

    args = argparser.parse_args()

   
    if args.create_folders:

        generation_config = open_json("generation_config.json")
        if not os.path.exists(generation_config["INPUTS_DIR"]):
            os.makedirs(generation_config["INPUTS_DIR"])
    
        if not os.path.exists(generation_config["OUTPUTS_DIR"]):
            os.makedirs(generation_config["OUTPUTS_DIR"])

    else:
        generator= Generator()
        generator.extend(args.inname, args.outname, args.nevents)




if __name__ == "__main__":
    main()
