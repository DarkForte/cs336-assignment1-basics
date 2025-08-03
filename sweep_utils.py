import torch
import numpy as np
import time
import train_utils
import utils
import transformer_blocks
import transformer_optimizer

def train_with_params(model_args, dataset, total_tokens, param_name, param_value):
    """
    Generic training function for parameter sweeps.
    
    Args:
        model_args: Namespace containing model parameters
        dataset: The dataset to train on
        total_tokens: Total number of tokens to process
        param_name: Name of the parameter being swept (for logging)
        param_value: Value of the parameter for this run
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # If we're sweeping a parameter, override it in the args
    if hasattr(model_args, param_name):
        setattr(model_args, param_name, param_value)
    
    transformer_model = transformer_blocks.Transformer(
        vocab_size=model_args.vocab_size,
        context_length=model_args.context_length,
        d_model=model_args.d_model,
        d_ff=model_args.d_ff,
        n_heads=model_args.n_heads,
        num_layers=model_args.num_layers,
        device=device
    )
    
    optimizer = transformer_optimizer.AdamW(
        transformer_model.parameters(),
        lr=model_args.lr if hasattr(model_args, 'lr') else 1e-4,
        betas=(model_args.beta1, model_args.beta2),
        weight_decay=model_args.weight_decay,
        eps=1e-8
    )
    
    losses = []
    tokens_processed = 0
    step = 0
    start_time = time.time()
    
    while tokens_processed < total_tokens:
        batch, labels = train_utils.get_batch(dataset, model_args.batch_size, model_args.context_length, device=device)
        
        optimizer.zero_grad()
        logits = transformer_model.forward(batch)
        loss = utils.cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        tokens_processed += model_args.batch_size * model_args.context_length
        step += 1
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            print(f"{param_name}: {param_value}, Tokens: {tokens_processed:,}, Steps: {step}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s, Tokens/sec: {tokens_per_sec:.1f}")
    
    checkpoint_name = f"transformer_checkpoint_{param_name}{param_value}.ckpt"
    train_utils.save_checkpoint(transformer_model, optimizer, tokens_processed, checkpoint_name)
    return losses
