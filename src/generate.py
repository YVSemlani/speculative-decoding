import torch
import torch.nn.functional as F

import numpy as np


def speculative_generate(draft_model, draft_tokenizer, teacher_model, teacher_tokenizer, prompt, **kwargs):

    MAX_TOKENS = kwargs.get('max_tokens', 1024)
    GAMMA = kwargs.get('gamma', 8)
    SAMPLE = kwargs.get('sample', False)
    KV_CACHE = kwargs.get('kv_cache', True)
    
    draft_cache, teacher_cache = None, None

    tokens_generated = 0
    full_text = prompt

    # Get EOS token id for both tokenizers
    draft_eos_token_id = draft_tokenizer.eos_token_id
    teacher_eos_token_id = teacher_tokenizer.eos_token_id

    while tokens_generated < MAX_TOKENS: # keep generating until we've hit the max tokens or EOS token
        draft_inputs = draft_tokenizer([prompt], return_tensors="pt").to(draft_model.device) # tokenize prompt for the draft model

        prompt_length = draft_inputs['input_ids'].shape[-1]

        # generate draft tokens
        draft_output = draft_model(**draft_inputs, do_sample=SAMPLE, max_new_tokens=GAMMA, return_dict_in_generate=True, output_scores=True, past_key_values=draft_cache, use_cache=KV_CACHE)
        draft_tokens = draft_output.sequences[0][prompt_length:].tolist()
        draft_text = draft_tokenizer.decode(draft_tokens)
        draft_logits = torch.stack(draft_output.scores, dim=1)
        draft_logprobs = F.log_softmax(draft_logits, dim=-1)

        # update cache
        draft_cache = draft_output.past_key_values

        # print draft output in green
        draft_text_clean = draft_text.replace('\n', '\\n')  # handle newlines for cleaner display
        print(f"\033[92m{draft_text_clean}\033[0m", end='', flush=True)

        # create input for teacher model
        teacher_inputs = teacher_tokenizer([prompt + draft_text], return_tensors="pt").to(teacher_model.device)

        # get teacher log probs
        teacher_output = teacher_model(**teacher_inputs, do_sample=SAMPLE, past_key_values=teacher_cache, use_cache=KV_CACHE)
        teacher_logits = teacher_output.logits
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)

        accepted_tokens = [] # store accepted tokens

        # accept or reject draft tokens
        eos_found = False
        reject = False
        for token_idx in range(len(draft_tokens)):
            current_token = draft_tokens[token_idx]

            teacher_logit_pos = prompt_length + token_idx - 1 # draft token is compared to step that generated it
            teacher_prob = teacher_logprobs[0, teacher_logit_pos, current_token]
            draft_prob = draft_logprobs[0, token_idx, current_token]

            if teacher_prob >= draft_prob: # p(x) >= q(x) -> accept
                accepted_tokens.append(current_token)
            else: # fallback to probabilistic acceptance
                prob = torch.exp(teacher_prob - draft_prob) # convert from log space to normal space to compare probs

                if np.random.uniform(0, 1) < prob: # accept with prob p(x)/q(x)
                    accepted_tokens.append(current_token)
                else: # reject -> sample token from residual distribution p(x) - q(x) -> exit
                    residual_prob = torch.clamp(torch.exp(teacher_logprobs[0, teacher_logit_pos, :]) - torch.exp(draft_logprobs[0, token_idx, :]), min=0) # p(x) - q(x) and clamp at 0 so we don't get negative probabilities
                    current_token = torch.multinomial(residual_prob, 1).item() # sample token from residual distribution

                    accepted_tokens.append(current_token)
                    reject = True # indicate that we rejected and sampled a token from the residual distribution

            if current_token == draft_eos_token_id or current_token == teacher_eos_token_id: eos_found = True # check if we just looked at the EOS token
            if eos_found or reject: break # exit generation loop if the EOS token is found or we rejected and sampled a token from the residual distribution
        
        # Erase the draft tokens and replace with accepted tokens
        accepted_text = draft_tokenizer.decode(accepted_tokens)
        draft_display_len = len(draft_text_clean)
        
        # Move cursor back and overwrite with accepted tokens
        print('\r' + ' ' * draft_display_len + '\r', end='', flush=True)  # clear the line
        print(accepted_text, end='', flush=True)  # print accepted tokens without color

        # update prompt with accepted tokens
        full_text += accepted_text
        tokens_generated += len(accepted_tokens)
        prompt = full_text

        # End generation if the EOS token is found
        if eos_found:
            break



