import torch
from pathlib import Path
import json
from tqdm import tqdm
import time
import os
import logging
from datetime import datetime
import argparse
import shutil

from data_pipeline import Config, TokenizerManager, DatasetPreprocessor
from model import create_model

def setup_logging(log_dir):
    """Set up logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def copy_data_to_local(data_path, logger):
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"Data path {data_path} does not exists")
        return None
    tmp_dir = Path("/tmp") / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying data from {data_path} to {tmp_dir}...")

    if data_path.is_dir():
        for item in data_path.glob('*'):
            dst = tmp_dir / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            elif item.is_dir():
                shutil.copytree(item, dst)
    else:
        dst = tmp_dir / data_path.name
        shutil.copy2(data_path, dst)
        logger.info(f"Copied {data_path.name}")
    
    logger.info(f"Data copied successfully to {tmp_dir}")
    return tmp_dir

def open_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def run_training_loop(
        train_loader,
        val_loader,
        model,          
        num_epochs, 
        learning_rate, 
        weight_decay, 
        save_every_n_epochs,
        test_loader = None,
        save_every_n_steps=10000,
        eval_log_step=1000,
        output_dir=None,
        seed=42,
        device=None,
        logger=None,
        ):
    
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create output directory if specified
    #make sure output dir exists
    print(f'Output dir exists and is located in: {output_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    start_time = time.time()
    total_steps = 0
    
    logger.info(f"Starting training with {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        # Wrap loader with tqdm for progress bar
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for step, batch in enumerate(train_iter):
            # Zero gradients
            optimizer.zero_grad()
            
            # Move data to device
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            # Update tracking
            total_loss += loss.item()
            total_steps += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_iter.set_postfix({"loss": loss.item()})
            
            # Save checkpoint if needed
            if save_every_n_steps and total_steps % save_every_n_steps == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{total_steps}"
                model.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint at step {total_steps} to {checkpoint_path}")
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Validation phase
        model.eval()
        total_eval_loss = 0
        
        # Wrap loader with tqdm for progress bar
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter):
                # Move data to device
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                
                # Update tracking
                total_eval_loss += loss.item()

                val_iter.set_postfix({"loss": loss.item()})

                # Calculate average validation loss
                if eval_log_step and batch_idx % eval_log_step == 0:
                    # Update progress bar
                    avg_eval_loss = total_eval_loss / len(val_loader)
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_eval_loss:.4f}")
                
        # Save the model every n epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            epoch_dir = output_dir / f"epoch_{epoch+1}"
            model.save_pretrained(epoch_dir)
            logger.info(f"Model saved at epoch {epoch+1} to {epoch_dir}")
        
        # Save best model
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            best_model_path = output_dir / "best_model"
            model.save_pretrained(best_model_path)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save_pretrained(final_model_path)
    
    # Log training summary
    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.2f}s")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved to {final_model_path}")
    
    return {
        "best_val_loss": best_val_loss,
        "final_model_path": final_model_path,
        "training_time": total_time
    }


# Set paths and device
def train(to_tmp, train_config_path, data_config_path):

    # Set paths and device
    train_config = open_json(train_config_path)
    data_config = Config.load_from_file(Path(data_config_path))
    
    paths = train_config["paths"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(timestamp+"_"+paths["output_dir"])
    output_dir = data_config.outputs_dir/"model_data"/output_dir

    log_dir = Path(timestamp+"_"+ paths["log_dir"])
    log_dir = data_config.outputs_dir/"model_data"/log_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    
    logger = setup_logging(log_dir)
    logger.info("Training started")
    logger.info(f"Config loaded: {train_config}")


    device = torch.device("cuda")
    logger.info(f"cuda available: {torch.cuda.is_available()}")
    
    
    logger.info(f"Using device: {device}")

    logger.info("Loading tokenizer...")

    #Load the tokenizer from the config file
    data_config.tokenizer_dir = Path(train_config["paths"]["tokenizer_dir"])
    data_config.tokenizer_dir = data_config.outputs_dir/"tokenizers"/data_config.tokenizer_dir

    data_config.chunked_dir = Path(train_config["paths"]["tokenizer_dir"] + "/chunked")
    data_config.chunked_dir = data_config.outputs_dir/"tokenizers"/data_config.chunked_dir


    data_path = data_config.chunked_dir

    if to_tmp and data_path:
        new_data_path = copy_data_to_local(data_path, logger)
        data_config.chunked_dir = new_data_path
    
    print(f"Data is loaded from {data_config.chunked_dir}")

    token_manager = TokenizerManager(config=data_config)
    tokenizer = token_manager.load_tokenizer(data_config.tokenizer_dir/"trained_tokenizer.json")

    logger.info("Loading dataset...")
    datasetpreprocessor = DatasetPreprocessor(config=data_config)
    dataset = datasetpreprocessor.load_chunked_dataset(tokenizer = tokenizer)
    
    dataset_params = train_config.get("dataset", {})
    train_ratio = dataset_params.get("train_ratio", 0.9)
    val_ratio = dataset_params.get("val_ratio", 0.1)
    test_ratio = dataset_params.get("test_ratio", 0.0)
    batch_size = dataset_params.get("batch_size", 1)

    train_loader, val_loader, _ = datasetpreprocessor.create_data_loaders(
        dataset = dataset, tokenizer = tokenizer, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio
    )
    
    logger.info(f"Dataset loaded: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create the model (you can easily swap this with another model in the future)
    model_name = train_config["model_name"]
    model_params = train_config["model_params"]

    model = create_model(tokenizer, model_name, model_params)
   
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    #Training loop fuckit
    logger.info("Starting training...")
    training_params = train_config["training"]

    training_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "device": device,
        "logger": logger,
        "output_dir": output_dir
    })

    # copying the configs and the tokenizer to the model folder
    shutil.copy2(data_config_path, output_dir.parent/data_config_path)
    shutil.copy2(train_config_path, output_dir.parent/train_config_path)
    shutil.copy2(data_config.tokenizer_dir/"trained_tokenizer.json", output_dir.parent/"trained_tokenizer.json")

    run_training_loop(**training_params)
    # Save the model
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with optional data localization")
    parser.add_argument('--data_to_tmp', action='store_true',
                        help='Copy data to /tmp for fast triton access')
    parser.add_argument('--train_config', type=str, help="Path to training configuration file")
    parser.add_argument('--data_config', type=str, help="Path to the data&tokenizer config")
    args = parser.parse_args()
    train(to_tmp = args.data_to_tmp, train_config_path = args.train_config, data_config_path = args.data_config)

