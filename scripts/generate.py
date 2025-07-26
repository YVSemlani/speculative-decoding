import sys
import argparse
import torch
import yaml

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from mlx_lm import load

from src.generate import speculative_generate

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # enable hf transfer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # load models and tokenizers
    draft_tokenizer = AutoTokenizer.from_pretrained(config['draft_model'], device_map="auto", torch_dtype=torch.bfloat16)
    draft_model = AutoModelForCausalLM.from_pretrained(config['draft_model'], device_map="auto", torch_dtype=torch.bfloat16)

    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'], device_map="auto", torch_dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(config['teacher_model'], device_map="auto", torch_dtype=torch.bfloat16)

    # remove sampling arguments if we're not sampling
    if not config.get('sample', False):
        draft_model.generation_config.do_sample = False
        draft_model.generation_config.temperature = None
        draft_model.generation_config.top_p = None
        draft_model.generation_config.top_k = None

        teacher_model.generation_config.do_sample = False
        teacher_model.generation_config.temperature = None
        teacher_model.generation_config.top_p = None
        teacher_model.generation_config.top_k = None

    # ask user for prompt
    prompt = input("Enter a prompt --> ")

    # add to the chat template
    prompt = [
        {"role": "user", "content": prompt}
    ]

    prompt = draft_tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=config.get('enable_thinking', False)
    )

    # drop draft and teacher model from the config we pass to avoid double assign
    config.pop('draft_model')
    config.pop('teacher_model')

    # generate
    speculative_generate(draft_model, draft_tokenizer, teacher_model, teacher_tokenizer, prompt, **config)