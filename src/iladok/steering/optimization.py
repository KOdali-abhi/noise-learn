import torch
from torch.optim import AdamW
from tqdm import tqdm

def optimize_vector_loop(model, tokenizer, vector_name, data, epochs, device):
    """
    Standard optimization loop for the steering vector.
    Data format: List of (input, output) tuples.
    """
    try:
        model.set_adapter(vector_name)
    except Exception:
        # If the wrapper does not support set_adapter, proceed assuming adapters are enabled.
        pass

    model.train()
    
    # Collect steering parameters (heuristic: names containing 'lora' or 'adapter' are likely steering params)
    steering_weights = [p for n, p in model.named_parameters() if ("lora" in n.lower() or "adapter" in n.lower()) and p.requires_grad]
    if not steering_weights:
        # Fallback: any parameter that requires grad
        steering_weights = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(steering_weights, lr=1e-4)
    
    # Simple data collator
    inputs = []
    for src, tgt in data:
        text = f"{src}\n{tgt}"
        inputs.append(text)
        
    encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    
    print(f"[Iladok] Optimizing vector '{vector_name}' on {len(data)} examples...")
    for epoch in range(epochs):
        # Basic single-step per epoch on the whole batch (suitable for tiny datasets)
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  > Epoch {epoch+1}: Signal Loss {loss.item():.4f}")
        
    model.eval()
    return model