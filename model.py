"""
Model definitions for music generation
"""

from pathlib import Path
from transformers import TransfoXLConfig, TransfoXLLMHeadModel

def create_model(tokenizer, model_name, model_params):
    """
    Create a model based on the specified name and parameters
    Args:
        model_name: Name of the model to create
        model_params: Parameters for the model
    Returns:
        model: The created model instance
    """
    config = {
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer["EOS_None"],
    }

    #add the model_params to config
    if model_params is not None:
        config.update(model_params)
    else:
        print("No model parameters provided, exiting")
        raise ValueError("No model parameters provided")

    
    if model_name == "transformer_xl":
        return create_transformer_xl_model(config)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def create_transformer_xl_model(train_config):
    """
    Create a Transformer-XL model for music generation
    
    Args:
        tokenizer: The tokenizer instance with vocabulary size information
        config_override: Optional dictionary with parameters to override defaults
    
    Returns:
        Configured TransfoXLLMHeadModel instance
    """

    if train_config["cutoffs"] == "[]":
        train_config["cutoffs"] = []
    trans_config = TransfoXLConfig(**train_config)


    print(trans_config)
    # Create and return the model
    return TransfoXLLMHeadModel(trans_config)