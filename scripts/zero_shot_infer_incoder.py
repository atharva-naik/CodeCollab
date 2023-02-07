import json
import torch
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

# don't tokenize instances
class UnTokenizedDataset(InCoderCellSeqDataset):
    def __init__(self, *args, **kwargs):
        super(UnTokenizedDataset, self).__init__(*args, **kwargs)
        self.ind_to_code_type = {v: k for k,v in self.code_type_to_ind.items()}

    def __getitem__(self, i: int):
        label = self.ind_to_code_type[self.data[i][1]]
        return [self.data[i][0], label]

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
                                 temperature: float=0.2, device: str="cuda",
                                 max_to_generate: int=128, stop_words=None):
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
        outputs = model.generate(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            max_length=max_length, stopping_criteria=stopping_criteria, 
            do_sample=True, top_p=0.95, temperature=temperature
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
    next_preds = []
    for hypo_str in hypo_strs:
        # to get rid of any errant ending tags at the start.
        hypo_str = hypo_str.replace("</text>", "").replace("</code>", "") 
        hypo_str = hypo_str.strip()
        try: hypo_str = hypo_str.split()[0].strip()
        except: hypo_str = "ERROR"
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    next_cell_preds = defaultdict(lambda:0)
    incoder_to_juice_format = {"<text>": "markdown", "<cell>": "code"}
    ind_to_code_type = {v: k for k,v in dataset.code_type_to_ind.items()}
    matches = 0
    tot = 0
    # pbar = tqdm(dataloader)
    pbar = tqdm(dataset.data)
    open("./analysis/incoder_next_cell_type_juice_val_preds.jsonl", "w")
    # for batch in pbar:
    for input in pbar:
        # trues = list(batch[1])
        next_cell_type_true = ind_to_code_type[input[1]]
        next_cell_type_pred = predict_next_cell_type(model, tokenizer, input[0])
        pred_true_str = f"{next_cell_type_pred} {next_cell_type_true}"
        next_cell_type_pred = incoder_to_juice_format.get(next_cell_type_pred, 
                                                          next_cell_type_pred)
        next_cell_preds[next_cell_type_pred] += 1
        matches += int(next_cell_type_pred == next_cell_type_true)
        tot += 1
        with open("./analysis/incoder_next_cell_type_juice_val_preds.jsonl", "a") as f:
            dumped_rec = json.dumps({
                "pred": next_cell_type_pred, 
                "true": next_cell_type_true,
            })
            f.write(dumped_rec+"\n")
        pbar.set_description(f"{pred_true_str} acc: {(100*matches/tot):.2f}")
    print(f"acc: {matches/tot}")
