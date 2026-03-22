import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def calculate_perplexity(text: str) -> float:
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    
    score = torch.exp(loss).item()
    return round(score, 2)


