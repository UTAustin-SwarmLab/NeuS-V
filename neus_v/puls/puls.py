from llm import *
from openai import OpenAI
from prompts import *
import json
import os
import re

def clean_and_parse_json(raw_str):
    start = raw_str.find('{')
    end = raw_str.rfind('}') + 1
    json_str = raw_str[start:end]
    return json.loads(json_str)

def process_specification(specification, propositions):
    new_propositions = []
    for prop in propositions:
        prop_cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", prop)
        prop_cleaned = re.sub(r"\s+", "_", prop_cleaned)
        new_propositions.append(prop_cleaned)

    for original, new in zip(propositions, new_propositions):
        specification = specification.replace(original, f'"{new}"')

    replacements = {
        "AND": "&",
        "OR": "|",
        "UNTIL": "U",
        "ALWAYS": "G",
        "EVENTUALLY": "F",
        "NOT": "!"
    }
    for word, symbol in replacements.items():
        specification = specification.replace(word, symbol)

    return new_propositions, specification

def create_prompt(prompt, modes):
    full_prompt = header + stage1_intro

    for i, m in enumerate(modes, start=1):
        full_prompt += f"{i}. {mode_prompts[m]}\n\n"
    
    full_prompt += stage2_intro
    full_prompt += spec_gen_intro.format(n=len(modes))
    full_prompt += tl_instructions
    full_prompt += input_template.format(prompt)
    
    full_prompt += expected_output_header
    for i, m in enumerate(modes):
        full_prompt += mode_outputs[m]
        if i != len(modes) - 1:
            full_prompt += "\n\n"
    full_prompt += expected_output_footer

    return full_prompt

def PULS(prompt, modes, openai_key=None):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    client = OpenAI()
    llm = LLM(client)

    full_prompt = create_prompt(prompt, modes)
    llm_output = llm.prompt(full_prompt)
    parsed = clean_and_parse_json(llm_output)

    final_output = {}

    for key, value in parsed.items():
        if key.endswith("_spec"):
            base_key = key.replace("_spec", "")
            propositions = parsed.get(base_key, [])
            cleaned_props, processed_spec = process_specification(value, propositions)
            final_output[base_key] = cleaned_props
            final_output[key] = processed_spec
        else:
            final_output[key] = value

    return final_output
