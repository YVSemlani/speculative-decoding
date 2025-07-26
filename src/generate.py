from functools import cache
import torch
import torch.nn.functional as F

import numpy as np
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from utils.autoregressive import autoregressive_generate_step

torch.set_float32_matmul_precision('medium')

def speculative_generate(draft_model, teacher_model, tokenizer, prompt, **kwargs):
    MAX_TOKENS = kwargs.get('max_tokens', 128)
    GAMMA = kwargs.get('gamma', 8)

    DEVICE = draft_model.device

    tokenized_prompt = tokenizer([prompt], return_tensors="pt")
    prompt_length = tokenized_prompt['input_ids'].shape[-1]

    input_ids = torch.full((1, prompt_length + MAX_TOKENS), tokenizer.pad_token_id, device=DEVICE)
    input_ids[0, :prompt_length] = tokenized_prompt['input_ids'][0] # fill the prompt into the input_ids

    draft_cache = DynamicCache()
    teacher_cache = DynamicCache()

    tokens_generated = 0
    current_position = prompt_length # start the current position at the end of the prompt

    start_time = time.perf_counter()
    with torch.inference_mode():
        while tokens_generated < MAX_TOKENS: # generate tokens until we reach the max number of tokens
            GAMMA = min(GAMMA, MAX_TOKENS - tokens_generated)

            draft_probs = torch.zeros((1, GAMMA, draft_model.config.vocab_size)).to(DEVICE)
            for tok_proposed in range(GAMMA): # generate GAMMA tokens from the draft model
                draft_output = draft_model(input_ids=input_ids[:, :current_position + tok_proposed], past_key_values=draft_cache, use_cache=True) # run model forward pass

                logits = F.softmax(draft_output.logits[:, -1, :], dim=-1) # softmax logits to get probabilities
                draft_probs[0, tok_proposed, :] = logits
                input_ids[0, current_position + tok_proposed] = torch.argmax(logits, dim=-1) # greedy decoding

                draft_cache = draft_output.past_key_values

            # run in parallel through the teacher model and get logits
            teacher_logits = teacher_model(input_ids[:, :current_position + GAMMA], past_key_values=teacher_cache, use_cache=True)

            # update the teacher cache
            teacher_cache = teacher_logits.past_key_values

            # slice the teacher logits to the last GAMMA tokens
            teacher_logits = teacher_logits.logits[:, current_position-1:current_position+GAMMA-1, :] # we need to be off by one

            # softmax the teacher logits
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # compute accepted draft positions
            r = torch.rand(GAMMA).to(DEVICE)
            probs = teacher_probs / draft_probs

            eos_token_found = False
            for gamma_idx in range(GAMMA):
                if r[gamma_idx] < probs[0, gamma_idx, input_ids[0, current_position + gamma_idx]]: # accept the draft token
                    if input_ids[0, current_position + gamma_idx] == tokenizer.eos_token_id:
                        eos_token_found = True
                        print("EOS TOKEN FOUND")
                        break
                else: # reject
                    mod_teacher_probs = torch.clamp(teacher_probs[0, gamma_idx, :] - draft_probs[0, gamma_idx, :], min=0)
                    mod_teacher_probs = mod_teacher_probs / mod_teacher_probs.sum()
                    teacher_token = torch.argmax(mod_teacher_probs, dim=-1) # replace 
                    input_ids[0, current_position + gamma_idx] = teacher_token

                    if teacher_token == tokenizer.eos_token_id:
                        eos_token_found = True
                        print("EOS TOKEN FOUND")
                    break

            input_ids[0, current_position + gamma_idx + 1:current_position + GAMMA] = tokenizer.pad_token_id
            
            # decode and print the new text
            print(tokenizer.decode(input_ids[0, current_position:current_position + gamma_idx + 1]), end="", flush=True)

            tokens_generated += gamma_idx + 1
            current_position += gamma_idx + 1

            tokens_to_discard = GAMMA - gamma_idx + 1

            # crop the cache
            for layer in range(len(draft_cache)):
                draft_cache.key_cache[layer] = draft_cache.key_cache[layer][:, :, :-tokens_to_discard, :]
                draft_cache.value_cache[layer] = draft_cache.value_cache[layer][:, :, :-tokens_to_discard, :]
                draft_cache._seen_tokens -= tokens_to_discard 
            
            for layer in range(len(teacher_cache)):
                teacher_cache.key_cache[layer] = teacher_cache.key_cache[layer][:, :, :-tokens_to_discard, :]
                teacher_cache.value_cache[layer] = teacher_cache.value_cache[layer][:, :, :-tokens_to_discard, :]
                teacher_cache._seen_tokens -= tokens_to_discard

            if eos_token_found:
                break

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Tokens per second: {tokens_generated / (end_time - start_time)}")

    return


if __name__ == "__main__":
    draft_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", device_map="auto")

    # Disable gradients for inference
    draft_model.requires_grad_(False)
    teacher_model.requires_grad_(False)

    params = {
        "max_tokens": 128,
        "gamma": 16,
    }

    prompt = [
        {"role": "user", "content": "Tell me about UNC Chapel Hill"}
    ]

    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print(prompt)
    speculative_generate(draft_model, teacher_model, tokenizer, prompt, **params)
        