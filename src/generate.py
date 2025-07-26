import torch
import torch.nn.functional as F

import numpy as np
import time

from transformers.cache_utils import DynamicCache

def speculative_generate(draft_model, draft_tokenizer, teacher_model, teacher_tokenizer, prompt, **kwargs):

    MAX_TOKENS = kwargs.get('max_tokens', 1024)
    GAMMA = kwargs.get('gamma', 8)
    SAMPLE = kwargs.get('sample', False)
    KV_CACHE = kwargs.get('kv_cache', True)

    COLLECT_STATS = kwargs.get('collect_stats', False)
    
    draft_kv_cache = DynamicCache()
    teacher_kv_cache = DynamicCache()

    tokens_generated = 0

    # Get EOS token id for both tokenizers
    draft_eos_token_id = draft_tokenizer.eos_token_id
    teacher_eos_token_id = teacher_tokenizer.eos_token_id

    if COLLECT_STATS:
        start_time = time.perf_counter()

    while tokens_generated < MAX_TOKENS: # keep generating until we've hit the max tokens or EOS token
        if tokens_generated == 0:
            draft_inputs = draft_tokenizer([prompt], return_tensors="pt").to(draft_model.device) # tokenize prompt for the draft model
            prompt_length = draft_inputs['input_ids'].shape[-1]
        else: 
            new_draft_inputs = draft_tokenizer([accepted_text], return_tensors="pt").to(draft_model.device)
            draft_inputs['input_ids'] = torch.cat([draft_inputs['input_ids'], new_draft_inputs['input_ids']], dim=1)
            draft_inputs['attention_mask'] = torch.cat([draft_inputs['attention_mask'], new_draft_inputs['attention_mask']], dim=1)
        
        # generate draft tokens
        draft_output = draft_model.generate(**draft_inputs, do_sample=SAMPLE, max_new_tokens=GAMMA, return_dict_in_generate=True, output_scores=True, past_key_values=draft_kv_cache)
        
        # Use the input length instead of prompt_length + tokens_generated
        input_length = draft_inputs['input_ids'].shape[-1]
        draft_tokens = draft_output.sequences[:, input_length:]
        draft_text = draft_tokenizer.decode(draft_tokens[0])
        draft_logits = torch.stack(draft_output.scores, dim=1)
        draft_probs = F.softmax(draft_logits, dim=-1)

        # update cache
        draft_kv_cache = draft_output.past_key_values

        # print draft output in green
        draft_text_clean = draft_text.replace('\n', '\\n')  # handle newlines for cleaner display
        print(f"\033[92m{draft_text_clean}\033[0m", end='', flush=True)

        # create input for teacher model
        teacher_inputs = draft_inputs
        teacher_inputs['input_ids'] = torch.cat([teacher_inputs['input_ids'], draft_tokens], dim=1)
        teacher_inputs['attention_mask'] = torch.cat([teacher_inputs['attention_mask'], torch.ones_like(draft_tokens)], dim=1)

        # get teacher log probs
        teacher_output = teacher_model(**teacher_inputs, do_sample=SAMPLE, past_key_values=teacher_kv_cache)
        teacher_logits = teacher_output.logits[:, input_length-1:-1, :]
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # update cache
        teacher_kv_cache = teacher_output.past_key_values

        accepted_tokens = [] # store accepted tokens

        # accept or reject draft tokens
        eos_found = False
        reject = False
        for token_idx in range(len(draft_tokens[0])):
            current_token = draft_tokens[0][token_idx]
            teacher_prob = teacher_probs[0, token_idx, current_token]
            draft_prob = draft_probs[0, token_idx, current_token]

            if teacher_prob >= draft_prob: # p(x) >= q(x) -> accept
                accepted_tokens.append(current_token)
            else: # fallback to probabilistic acceptance
                prob = teacher_prob / draft_prob # p(x)/q(x)

                if np.random.uniform(0, 1) < prob: # accept with prob p(x)/q(x)
                    accepted_tokens.append(current_token)
                else: # reject -> sample token from residual distribution p(x) - q(x) -> exit
                    residual_prob = torch.clamp(teacher_probs[0, token_idx, :] - draft_probs[0, token_idx, :], min=0) # p(x) - q(x) and clamp at 0 so we don't get negative probabilities
                    current_token = torch.multinomial(residual_prob, 1).item() # sample token from residual distribution

                    accepted_tokens.append(current_token)
                    reject = True # indicate that we rejected and sampled a token from the residual distribution

            if current_token == draft_eos_token_id or current_token == teacher_eos_token_id: eos_found = True # check if we just looked at the EOS token
            if eos_found or reject: break # exit generation loop if the EOS token is found or we rejected and sampled a token from the residual distribution
        
        # Erase the draft tokens and replace with accepted tokens
        accepted_text = draft_tokenizer.decode(accepted_tokens)
        draft_display_len = len(draft_text_clean)
        
        # Move cursor back and overwrite with accepted tokens
        print('\b' * draft_display_len, end='', flush=True)  # backspace to erase only the green text
        print(accepted_text, end='', flush=True)  # print accepted tokens without color

        # update prompt with accepted tokens
        tokens_generated += len(accepted_tokens)

        # crop the cache to include only the accepted tokens + prompt
        draft_kv_cache = draft_kv_cache.crop(prompt_length + tokens_generated)
        teacher_kv_cache = teacher_kv_cache.crop(prompt_length + tokens_generated)

        # End generation if the EOS token is found
        if eos_found:
            break

    if COLLECT_STATS:
        end_time = time.perf_counter()
        print(f"\nTime taken: {end_time - start_time} seconds")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Tokens per second: {tokens_generated / (end_time - start_time)}")

    return accepted_text

