# sample some data for annotating as situated or not situated.
import os
import re
import sys
import json
import random
import string
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datautils import camel_case_split
# from datautils.keyphrase_extraction import EnsembleKeyPhraseExtractor

# seed for determinism
random.seed(2023)

def has_both_num_and_letter(word: str) -> bool:
    has_letter = False
    has_digit = False
    for char in word:
        if char in string.ascii_uppercase+string.ascii_lowercase: has_letter = True
        if char in string.digits: has_digit = True
        if has_letter and has_digit: return True
    return False

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_step_numbers(text):
    clean_text = []
    regex = r'^\D*\d+\D*(\.)?\D*$'
    for word in text.split():
        clean_text.append(re.sub(regex, '', word.strip()))

    return " ".join(clean_text).strip()

def process_step(step: str):
    step = step.strip()
    step = step.replace("➤", "")
    step = step.replace("&nbsp;", "")
    for punct in ["!", "-", ",", "'", '"', "#", "$", 
                  "%", "&", ";", "\x08", ")", "(",
                  "\\", "|", "^", "~", "`", "=",
                  "[", "]", "{", "}", "_", "+", 
                  "-", "/", "*", ":", "?"]:
        step = step.replace(punct," ")
    step = remove_step_numbers(step)
    step = step.replace(".", " ")
    step = remove_html_tags(step)
    step = step.replace("<", " ").replace(">", " ")
    # split and re-join for uniform spaces.
    word_seq = []
    for word in step.split():
        if not(word.replace('.','',1).isdigit() or has_both_num_and_letter(word)):
            word_seq += camel_case_split(word)

    return " ".join(word_seq).lower()

# main
if __name__ == "__main__":
    kb_path = sys.argv[1] # get the path.
    try: k = int(sys.argv[2])
    except IndexError: k = 1000
    KB = json.load(open(kb_path))

    step_equivalence_map = defaultdict(lambda:set())
    for path_key in tqdm(KB):
        for step in path_key.split("->"): 
            step_equivalence_map[process_step(step)].add(step) 
    step_equivalence_map = {k: v for k, v in sorted(step_equivalence_map.items())}
    pool_of_steps = []
    for step_class, step_class_member_list in tqdm(step_equivalence_map.items()):
        for raw_step in step_class_member_list:
            pool_of_steps.append({
                "raw_step": raw_step, 
                "processed_step": step_class,
                "is_situated": "",
            })
    # sample 1k instances.
    # sampled_steps = []
    sampled_steps = random.sample(pool_of_steps, k=k)
    # for ind in sampled_indices: sampled_steps.append(pool_of_steps[ind])
    annot_df = pd.DataFrame(sampled_steps)
    annot_df.to_csv("./juice_train_KB_situated_step_annot.csv", index=False)