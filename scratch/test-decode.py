### Testing the flow of spec decoding without differentiated draft and target
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import (
    DynamicCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

import numpy as np
import time

draft_model_name = "Qwen/Qwen3-0.6B"
teacher_model_name = "Qwen/Qwen3-4B"
GAMMA = 8 # how many tokens the draft model generates at once
MAX_TOKENS = 128


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(draft_model_name) # assumes the tokenizer is the same
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# prepare the draft input
prompt = "Give me a primer on stochastic calculus"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)

# Clear any default generation config that might have these parameters
draft_model.generation_config.do_sample = False
draft_model.generation_config.temperature = None
draft_model.generation_config.top_p = None
draft_model.generation_config.top_k = None

teacher_model.generation_config.do_sample = False
teacher_model.generation_config.temperature = None
teacher_model.generation_config.top_p = None
teacher_model.generation_config.top_k = None

# initialize the cache
draft_cache = DynamicCache()
teacher_cache = DynamicCache()

start_time = time.perf_counter()
tokens_generated = 0
while tokens_generated < MAX_TOKENS:
    draft_inputs = tokenizer([text], return_tensors="pt").to(draft_model.device)
    prompt_length = draft_inputs['input_ids'].shape[-1]
    if tokens_generated == 0:
        print(text, end='', flush=True)
    #print(f"Text = {prompt_length} tokens | # Tokens generated = {tokens_generated}")

    # conduct text completion w/ draft model
    draft_output = draft_model.generate(
        **draft_inputs,
        do_sample=False,
        return_dict_in_generate=True,
        max_new_tokens=GAMMA,
        output_scores=True,
        use_cache=True,
        cache_implementation="static"
    )

    generated_ids = draft_output.sequences
    draft_logits = torch.stack(draft_output.scores, dim=1)
    draft_logprobs = F.log_softmax(draft_logits, dim=-1)

    draft_tokens = generated_ids[0][len(draft_inputs.input_ids[0]):].tolist()
    draft_text = tokenizer.decode(draft_tokens)
    #print(f"Draft model generated: {draft_text}")

    # prepare the teacher input
    teacher_text = text + draft_text
    teacher_inputs = tokenizer([teacher_text], return_tensors="pt").to(teacher_model.device)

    # get log probs with teacher model
    teacher_output = teacher_model(**teacher_inputs, do_sample=False, use_cache=True, cache_implementation="static")
    teacher_logprobs = F.log_softmax(teacher_output.logits, dim=-1)


    accepted_tokens = []
    for token_idx in range(len(draft_tokens)):
        teacher_logit_pos = prompt_length + token_idx - 1 # the draft token is compared to the step that generated it rather than the step at token_idx
        current_token = draft_tokens[token_idx]
        
        teacher_prob = teacher_logprobs[0, teacher_logit_pos, current_token]
        draft_prob = draft_logprobs[0, token_idx, current_token]

        if teacher_prob >= draft_prob: # p(x) >= q(x) -> accept
            accepted_tokens.append(current_token)
            #print(f'    Draft token accepted initially')
        else: # evaluate with prob p(x)/q(x) and randomly sampled U
            prob = torch.exp(teacher_prob - draft_prob)
            if np.random.uniform(0, 1) < prob: # accept outcome
                accepted_tokens.append(current_token)
                #print(f'    Draft token accepted after p(x)/q(x) outcome')
            else: # reject outcome -> sample last token then exit

                # Why residual?
                # TO-DO: Implement Beam Search sampling
                teacher_prob = torch.exp(teacher_logprobs[0, teacher_logit_pos, :])
                draft_prob = torch.exp(draft_logprobs[0, token_idx, :])
                residual_probs = torch.clamp(teacher_prob - draft_prob, min=0)

                residual_probs = residual_probs / residual_probs.sum()  # Normalize
                selected_token_idx = torch.multinomial(residual_probs, 1).item()

                accepted_tokens.append(selected_token_idx)
                #print(f'    Draft token rejected; Sampled token {selected_token_idx} from Teacher')
                break # stop sampling when rejected
        
        token_idx += 1

        
    accepted_text = tokenizer.decode(accepted_tokens)

    # add accepted to the original text and update our tokens generated
    text += accepted_text
    tokens_generated += len(accepted_tokens)
    print(accepted_text, end='', flush=True)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nGeneration completed with {tokens_generated / elapsed_time} tok/s")

