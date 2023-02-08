import json
import torch
import random
import numpy as np
from typing import *
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from datautils.dataloaders import InCoderCellSeqDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria

# special tokens.
PAD = "<pad>"
# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

# seed using random, numpy, torch.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# all the code for batched usage sourced from here: https://github.com/dpfried/incoder/blob/main/example_batched_usage.py
class StopWordsStoppingCriteria(StoppingCriteria):
    """class for stop words based stopping criterion
    """
    def __init__(self, init_lengths: List[int], stop_words_encoded: List[List[int]]):
        super().__init__()
        self.init_lengths = init_lengths
        if stop_words_encoded is None:
            stop_words_encoded = []
        else:
            assert isinstance(stop_words_encoded[0], list)
        assert isinstance(stop_words_encoded, list)
        self.stop_words_encoded = stop_words_encoded

    def _contains_stop_words(self, tokens: List[int]):
        if not bool(self.stop_words_encoded):
            return False
        for start_ix in range(len(tokens)):
            for swe in self.stop_words_encoded:
                if tokens[start_ix:start_ix+len(swe)] == swe:
                    return True
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for init_length, i_tokens in zip(self.init_lengths, input_ids):
            if not self._contains_stop_words(i_tokens[init_length:].tolist()):
                return False
        return True

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def generate(input: str, max_to_generate: int=128, 
             temperature: float=0.2, device: str="cuda"):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if device == "cuda": input_ids = input_ids.cuda()
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str

def infill(parts: List[str], max_to_generate: int=128, 
           temperature: float=0.2, extra_sentinel: bool=True, 
           max_retries: int=1, verbose: bool=True):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.
    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in 
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.
    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        if verbose: print(f"retry {retries_attempted}")
        
        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
        
        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate(prompt, max_to_generate, temperature)
            completion = completion[len(prompt):]
            if EOM not in completion:
                if verbose: print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        print("generated text:")
        print(prompt)
        print()
        print("parts:")
        print(parts)
        print()
        print("infills:")
        print(infills)
        print()
        print("restitched text:")
        print(text)
        print()
    
    return {
        'text': text, # str, the completed document (with infills inserted)
        'parts': parts, # List[str], length N. Same as passed to the method
        'infills': infills, # List[str], length N-1. The list of infills generated
        'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
    } 

def predict_continuation(model, tokenizer, input: str, generator_args: dict={
        "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 100, }): #temperature: float=0.8):
    args = tokenizer(input, return_tensors="pt", add_special_tokens=True)
    i = len(args["input_ids"].squeeze())
    for k in args: args[k] = args[k].cuda()
    args.update(generator_args) # add the generator args to the tensor inputs.
    op = model.generate(**args).squeeze()

    return tokenizer.decode(op[i:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)

def remove_extra_code(input):
    """TODO: what does this function do?"""
    min_stop_position = len(input)
    stop_tokens = ["\nclass", "\ndef", "\n#", "\nif", "\nassert", "\nclass", "<|/ file"]
    for stop_token in stop_tokens:
        if stop_token in input: min_stop_position = min(min_stop_position, input.index(stop_token)) 
    return input[:min_stop_position]

def batched_predict_continuation(model, tokenizer, inputs: List[str], trim: bool=True,
                                 temperature: float=0.8, strategy: str="nucleus", top_p=0.95,
                                 device: str="cuda", max_to_generate: int=128, stop_words=None):
    assert tokenizer.padding_side == 'left'
    assert isinstance(inputs, list)
    batch = tokenizer(inputs, return_tensors="pt", 
                      truncation=True, padding="longest")
    batch = batch.to(device)
    max_input_length = batch.input_ids.size(1)
    max_length = max_input_length + max_to_generate
    stopping_criteria = StoppingCriteriaList()
    if stop_words is not None:
        stop_words_encoded = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
        stopping_criteria.append(StopWordsStoppingCriteria([max_input_length for l in inputs], stop_words_encoded))
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        if strategy == "nucleus":
            outputs = model.generate(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                max_length=max_length, stopping_criteria=stopping_criteria, 
                do_sample=True, top_p=top_p, temperature=temperature
            )
        elif strategy == "greedy": 
            outputs = model.generate(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                max_length=max_length, stopping_criteria=stopping_criteria, 
                do_sample=False, num_beams=1,
            )    

    hypo_strs = []
    for input, output in zip(inputs, outputs):
        detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
        while detok_hypo_str.startswith(PAD):
            detok_hypo_str = detok_hypo_str[len(PAD):]
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS):]

        if trim:
            detok_hypo_str = detok_hypo_str[len(input):]
            detok_hypo_str = remove_extra_code(detok_hypo_str)
        hypo_strs.append(detok_hypo_str)

    return hypo_strs

def batch_predict_next_cell_type(
    model, tokenizer, inputs: List[str], *args, **kwargs):
    hypo_strs = batched_predict_continuation(model, tokenizer, 
                                             inputs, *args, **kwargs)
    incoder_to_juice_format = {"<text>": "markdown", "<cell>": "code"}
    next_preds = []
    for hypo_str in hypo_strs:
        # to get rid of any errant ending tags at the start.
        hypo_str = hypo_str.replace("</text>", "").replace("</code>", "") 
        hypo_str = hypo_str.strip()
        try: hypo_str = hypo_str.split()[0].strip()
        except: hypo_str = "ERROR"
        hypo_str = incoder_to_juice_format.get(
            hypo_str, hypo_str
        )
        next_preds.append(hypo_str)
    
    return next_preds

def predict_next_cell_type(model, tokenizer, input):
    """zero shot prediction of the next cell type."""
    try:
        cont = predict_continuation(model, tokenizer, input) # continuation.
        return cont.strip().split()[0].strip()
    except IndexError as e:
        print(e)
        return "ERROR"

# main
if __name__ == "__main__":
    dataset = UnTokenizedDataset("./data/juice-dataset/dev.jsonl")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B") # InCoder tokenizer.
    tokenizer = dataset.tokenizer
    model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B") # incoder-1B model.
    model.cuda() # move to GPU
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    next_cell_preds = defaultdict(lambda:0)
    
    matches, tot = 0, 0
    pbar = tqdm(dataloader)
    WRITE_PATH = "./analysis/incoder_greedy_nctp_juice_val.jsonl"
    open(WRITE_PATH, "w")
    for batch in pbar:
        trues = list(batch[1])
        inputs = list(batch[0])
        # preds = batch_predict_next_cell_type(
        #     model, tokenizer, inputs,
        #     temperature=0.1, top_p=0.95,
        # )
        preds = batch_predict_next_cell_type(
            model, tokenizer, inputs,
            strategy="greedy",
        )
        with open(WRITE_PATH, "a") as f:
            for true, pred in zip(trues, preds): 
                next_cell_preds[pred] += 1
                matches += int(pred == true)
                tot += 1
                dumped_rec = json.dumps({"pred": pred, "true": true})
                f.write(dumped_rec+"\n")
        pbar.set_description(f"acc: {(100*matches/tot):.2f}")
    print(f"acc: {matches/tot}")