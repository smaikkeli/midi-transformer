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
    if model_name == "transformer_xl":
        return create_transformer_xl_model(tokenizer, config_override = model_params)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def create_transformer_xl_model(tokenizer, config_override=None):
    """
    Create a Transformer-XL model for music generation
    
    Args:
        tokenizer: The tokenizer instance with vocabulary size information
        config_override: Optional dictionary with parameters to override defaults
    
    Returns:
        Configured TransfoXLLMHeadModel instance
    """
    # Default configuration
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_embed": 256,
        "d_model": 512,
        "n_layer": 8,
        "n_head": 6,
        "mem_len": 256,
        "clamp_len": 0,
        "cutoffs": [],
        "adaptive": False,
        "eos_token_id": tokenizer["EOS_None"]
    }
    
    # Override with any provided configs
    if config_override is not None:
        model_config.update(config_override)
    
    # Create the configuration
    config = TransfoXLConfig(**model_config)

    print(config.eos_token_id)
    print(config.cutoffs)
    
    # Create and return the model
    return TransfoXLLMHeadModel(config)