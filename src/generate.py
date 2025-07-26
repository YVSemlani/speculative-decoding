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

    print(f"Using device: {DEVICE}")

    tokenized_prompt = tokenizer([prompt], return_tensors="pt")
    prompt_length = tokenized_prompt['input_ids'].shape[-1]

    input_ids = torch.full((1, prompt_length + MAX_TOKENS), tokenizer.pad_token_id, device=DEVICE)
    input_ids[0, :prompt_length] = tokenized_prompt['input_ids'][0] # fill the prompt into the input_ids

    # ---------------------------------------------------------------------
    # Prime KV caches with the full prompt so subsequent calls can send only
    # the *new* tokens.  Each model now has `prompt_length` tokens stored.
    # ---------------------------------------------------------------------
    with torch.no_grad():
        draft_prime   = draft_model(input_ids[:, :prompt_length], use_cache=True)
        teacher_prime = teacher_model(input_ids[:, :prompt_length], use_cache=True)

    draft_cache   = draft_prime.past_key_values
    teacher_cache = teacher_prime.past_key_values

    tokens_generated = 0
    current_position = prompt_length # start the current position at the end of the prompt

    # compile models
    draft_model.forward = torch.compile(draft_model.forward)
    teacher_model.forward = torch.compile(teacher_model.forward)

    start_time = time.perf_counter()
    with torch.no_grad():
        while tokens_generated < MAX_TOKENS: # generate tokens until we reach the max number of tokens
            GAMMA = min(GAMMA, MAX_TOKENS - tokens_generated)

            # keep only the probability of the chosen token for each proposal
            draft_sel_probs = torch.empty(GAMMA, device=DEVICE)
            draft_probs_rows = []   # full prob rows kept only for potential rejection
            for tok_proposed in range(GAMMA): # generate GAMMA tokens from the draft model
                # Send exactly ONE new token (keep rank by using a slice)
                draft_input = input_ids[:, current_position + tok_proposed - 1 : current_position + tok_proposed]

                draft_output = draft_model(
                    input_ids=draft_input,
                    past_key_values=draft_cache,
                    use_cache=True,
                )

                logits_row = F.softmax(draft_output.logits[:, -1, :], dim=-1)  # probabilities for this position
                next_token = torch.argmax(logits_row, dim=-1)

                # store scalar prob of the sampled token for fast accept-test
                draft_sel_probs[tok_proposed] = logits_row[0, next_token]

                # keep the full row only in case we need it after rejection
                draft_probs_rows.append(logits_row.squeeze(0))

                # write token to sequence
                input_ids[0, current_position + tok_proposed] = next_token

                draft_cache = draft_output.past_key_values

            # run in parallel through the teacher model and get logits
            teacher_input = input_ids[:, current_position - 1 : current_position + GAMMA]
            teacher_logits = teacher_model(
                teacher_input,
                past_key_values=teacher_cache,
                use_cache=True,
            )

            # update the teacher cache
            teacher_cache = teacher_logits.past_key_values

            # slice the teacher logits to the last GAMMA tokens
            #teacher_logits = teacher_logits.logits[:, current_position-1:current_position+GAMMA-1, :] # we need to be off by one

            # softmax the teacher logits (γ rows)
            teacher_probs = F.softmax(teacher_logits.logits[:, :-1, :], dim=-1)

            # probability of teacher for the same chosen tokens
            sel_teacher_probs = teacher_probs[0, torch.arange(GAMMA, device=DEVICE),
                                              input_ids[0, current_position:current_position + GAMMA]]

            # compute accepted draft positions (scalar ratio)
            r = torch.rand(GAMMA, device=DEVICE)
            probs_ratio = sel_teacher_probs / draft_sel_probs

            eos_token_found = False
            for gamma_idx in range(GAMMA):
                if r[gamma_idx] < probs_ratio[gamma_idx]:  # accept the draft token
                    if input_ids[0, current_position + gamma_idx] == tokenizer.eos_token_id:
                        eos_token_found = True
                        print("EOS TOKEN FOUND")
                        break
                else: # reject
                    # lazily compute the adjustment using the stored draft row
                    mod_teacher_probs = torch.clamp(
                        teacher_probs[0, gamma_idx, :] - draft_probs_rows[gamma_idx],
                        min=0
                    )
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

            accepted = gamma_idx + 1  # number of draft tokens kept this step

            discard_draft   = GAMMA - accepted          # only the rejected ones
            discard_teacher = discard_draft + 1         # plus duplicate prefix

            # crop the cache
            for layer in range(len(draft_cache)):
                if discard_draft:
                    draft_cache.key_cache[layer]   = draft_cache.key_cache[layer][:, :, :-discard_draft, :]
                    draft_cache.value_cache[layer] = draft_cache.value_cache[layer][:, :, :-discard_draft, :]
            draft_cache._seen_tokens -= discard_draft
            
            for layer in range(len(teacher_cache)):
                if discard_teacher:
                    teacher_cache.key_cache[layer]   = teacher_cache.key_cache[layer][:, :, :-discard_teacher, :]
                    teacher_cache.value_cache[layer] = teacher_cache.value_cache[layer][:, :, :-discard_teacher, :]
            teacher_cache._seen_tokens -= discard_teacher

            if eos_token_found:
                break

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Tokens per second: {tokens_generated / (end_time - start_time)}")

    return


if __name__ == "__main__":
    # Keep *all* weights on the single GPU to prevent host-to-device transfers.
    draft_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device_map={"": 0},            # everything on GPU 0
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        device_map={"": 0},            # everything on GPU 0
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        max_memory={0: "47GiB"},
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Disable gradients for inference
    draft_model.requires_grad_(False)
    teacher_model.requires_grad_(False)

    params = {
        "max_tokens": 128,
        "gamma": 8,
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

    # test manual generaiton and time
    with torch.no_grad():
        draft_model.eval()
        teacher_model.eval()

        start_time = time.perf_counter()
        tokenized_prompt = tokenizer([prompt], return_tensors="pt").to("cuda")
        out = teacher_model.generate(**tokenized_prompt, max_new_tokens=128, do_sample=False)
        print(tokenizer.decode(out[0], skip_special_tokens=True))
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Tokens per second: {128 / (end_time - start_time)}")

        
        
        
        
        