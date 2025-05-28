import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from transformers import TransfoXLLMHeadModel

import json
import argparse
import time

class TransXLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, mems):
        mems = list(torch.unbind(mems, dim = 0))
        pred, mems = self.model(input_ids = input_ids, mems = mems)
        mems = torch.stack(mems, dim = 0)
        return pred, mems
    
class GeneratorWrapper(nn.Module):
    def __init__(self, model_wrapper: TransXLWrapper, 
                 ids_tensor: torch.Tensor, 
                 mems_dummy: torch.Tensor):
        super().__init__()
        
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                self.model = torch.jit.trace(model_wrapper.eval(), 
                                    (ids_tensor, mems_dummy),
                                    )
           
        self.batch_size = 1
        self.n_layer = model_wrapper.model.transformer.n_layer
        self.mem_len = model_wrapper.model.transformer.mem_len
        self.d_model = model_wrapper.model.transformer.d_model
        self.vocab_size = model_wrapper.model.transformer.n_token

        #init mems
        self.init_mems = torch.zeros(self.n_layer, self.mem_len, self.batch_size, self.d_model, dtype=torch.float32)

        #Create tracker for repetition penalty
        self.token_counts = torch.zeros(self.batch_size, self.vocab_size, dtype=torch.int32)

    def _rep_penalty(self, logits: torch.Tensor, rep_penalty: float) -> torch.Tensor:
        """Apply repetition penalty - TorchScript friendly"""
        if rep_penalty != 1.0:
            penalty_mask = self.token_counts > 0
            # Use where instead of in-place operations for TorchScript compatibility
            penalized_logits = logits / rep_penalty
            logits = torch.where(penalty_mask, penalized_logits, logits)
        return logits

    def _top_k_sampling(self, logits, top_k: int):
        """Apply top-k filtering to logits"""
        if top_k > 0:
            # Get top-k values and indices
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            # Set all values below top-k to -inf
            logits[logits < top_k_values[:, -1]] = float('-inf')
        return logits

    def _sample_token(self, logits, top_k: int):
        logits = self._top_k_sampling(logits, top_k)
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        return next_token
    
    def forward(self, input_ids, 
                n_events:int, 
                temperature: float, 
                repetition_penalty: float,
                topk: int,
                ):
        
        generated_tokens = torch.zeros(self.batch_size, n_events, dtype=torch.long)

        # Initial prediction
        pred, mems = self.model(input_ids, self.init_mems) # [batch_size, seq_len, vocab_size]
        pred_logits = pred[:, -1, :] # Get logits for the last token

        for step in range(n_events):
            scaled_logits = pred_logits / temperature

            scaled_logits = self._rep_penalty(scaled_logits, repetition_penalty)

            next_token = self._sample_token(scaled_logits, topk)
            #next_token = torch.argmax(scaled_logits, dim=-1)
            generated_tokens[:, step] = next_token  
            batch_indices = torch.arange(self.batch_size)
            self.token_counts[batch_indices, next_token] += 1

            # Predict next logits, adding new token to input sequence
            pred, mems = self.model(next_token.unsqueeze(0), mems)  # Keep predicting for new sequence
            pred_logits = pred[:, -1, :]  # Get logits for the last token

        return generated_tokens
    
def create_gen_onnx_model(model_path: str,
                        output_name: str):
    model = TransfoXLLMHeadModel.from_pretrained(model_path, torchscript=True)
    config_path = "./best_model/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_wrapper = TransXLWrapper(model)
    ids_tensor = torch.zeros((1, 100), dtype=torch.long)  
    mems_dummy = torch.zeros((model_wrapper.model.transformer.n_layer, 
                              model_wrapper.model.transformer.mem_len, 
                              1, 
                              model_wrapper.model.transformer.d_model), 
                             dtype=torch.float32)  
    
    n_events = torch.tensor(10, dtype=torch.int64)
    temperature = torch.tensor(1.2, dtype=torch.float32)
    repetition_penalty = torch.tensor(0.9, dtype=torch.float32)
    topk = torch.tensor(10, dtype=torch.int64)

    gen_wrapper = GeneratorWrapper(model_wrapper, ids_tensor, mems_dummy)
    gen_wrapper.eval()
    gen_model = torch.jit.script(gen_wrapper,
                                (ids_tensor, n_events, temperature, repetition_penalty, topk))

    with torch.no_grad():
        torch.onnx.export(gen_model,
                        (ids_tensor, n_events, temperature, repetition_penalty, topk),
                        output_name,
                        input_names = ["input_ids", "n_events", "temperature", "repetition_penalty", "topk"],
                        output_names = ["gen_seq"],
                        dynamic_axes = {"input_ids": {1: "sequence"},
                                        "gen_seq": {1: "sequence"}},
                        strict=True,
                        do_constant_folding=True,
                        )
        
    
    onnx_model = onnx.load("gen.onnx")
    onnx.checker.check_model(onnx_model)


    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = output_name
    
    ort_session = ort.InferenceSession(output_name, sess_options=sess_options)

    print("Running inference with exported model")

    input_ids = ids_tensor.numpy()
    n_events = np.array(1000, dtype=np.int64)
    temperature = np.array(1.2, dtype=np.double)
    repetition_penalty = np.array(1.2, dtype=np.double)
    topk = np.array(10, dtype=np.int64)

    time_start = time.time()
    _ = ort_session.run(None, {
            "input_ids": input_ids,
            "n_events": n_events,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "topk": topk,
    })
    time_end = time.time()
    print(f"Inference time for {n_events} tokens: {time_end - time_start:.2f} seconds")

    
#Make main function that takes model path and output name as arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TransfoXL model to ONNX format")
    parser.add_argument("--model_path", type=str, required=True,  
                        help="Path to the TransfoXL model directory")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Output ONNX model file name")
    args = parser.parse_args()
    create_gen_onnx_model(args.model_path, args.output_name)

#example usage:
# conda activate midi-transformer
# python convert_to_onnx.py --model_path ./best_model --output_name gen.onnx
