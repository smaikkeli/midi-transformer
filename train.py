import torch
from pathlib import Path
import json
from tqdm import tqdm
import time
import os
import logging
from datetime import datetime

from data_pipeline import TokenizerManager, DatasetPreprocessor
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
        save_every_n_steps=None,
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
    if output_dir:
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
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update tracking
            total_loss += loss.item()
            total_steps += 1
            
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
            for batch in val_iter:
                # Move data to device
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                
                # Update tracking
                total_eval_loss += loss.item()
                
                # Update progress bar
                val_iter.set_postfix({"loss": loss.item()})
        
        # Calculate average validation loss
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
def train():
    # Set paths and device
    config = open_json("train_config.json")
    paths = config["paths"]
    output_dir = Path(paths["output_dir"])
    log_dir = Path(paths["log_dir"])

    #Output and log dir into path objects
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info("Training started")
    logger.info(f"Config loaded: {config}")

    # Save config copy with timestamp for reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_copy_path = output_dir / f"config_{timestamp}.json"
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_copy_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else 
                            "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load the tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = TokenizerManager.load_tokenizer()
    
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = DatasetPreprocessor.load_tokenized_dataset(tokenizer = tokenizer)
    
    dataset_params = config.get("dataset", {})
    train_ratio = dataset_params.get("train_ratio", 0.9)
    val_ratio = dataset_params.get("val_ratio", 0.1)
    test_ratio = dataset_params.get("test_ratio", 0.0)
    batch_size = dataset_params.get("batch_size", 4)

    # Split into train/validation sets
    train_loader, val_loader, _ = DatasetPreprocessor.create_data_loaders(
        tokenizer, dataset, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    
    logger.info(f"Dataset loaded: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Create the model (you can easily swap this with another model in the future)
    model_name = config["model_name"]
    model_params = config["model_params"]

    model = create_model(tokenizer, model_name, model_params)
   
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    #Training loop fuckit
    logger.info("Starting training...")
    training_params = config["training"]

    training_params.update({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "device": device,
        "logger": logger
    })

    run_training_loop(**training_params)
    
    
    # Save the model
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
if __name__ == "__main__":
    train()