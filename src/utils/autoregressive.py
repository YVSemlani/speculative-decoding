import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def autoregressive_generate_step(model, input_ids, past_key_values=None):
    out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True) # run model forward pass

    logits = F.softmax(out.logits[:, -1, :], dim=-1) # softmax logits to get probabilities
    next_token = torch.argmax(logits, dim=-1) # greedy decoding
    return next_token, logits, out.past_key_values
