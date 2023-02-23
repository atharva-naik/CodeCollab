import json
from typing import *
from tqdm import tqdm
from datautils import read_jsonl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datautils.markdown_cell_analysis import process_markdown

def get_bart_tl_topics(input, model, tokenizer):
    enc = tokenizer(input, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=15,
        min_length=1,
        do_sample=False,
        num_beams=25,
        length_penalty=1.0,
        repetition_penalty=1.5
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded

def generate_topics_for_each_nb(data: List[dict], model, 
                                tokenizer, path: str) -> List[dict]:
    generated_topics = []
    pbar = tqdm(data)
    open(path, "w")
    for rec in pbar:
        ctxt = rec["context"][::-1]
        markdowns = []
        for cell in ctxt:
            # pbar.set_description(f"{i}/{len(ctxt)}")
            if cell["cell_type"] == "markdown":
                proc_md_cell = process_markdown(cell["nl_original"])
                markdowns.append(proc_md_cell)
        topic = get_bart_tl_topics(
            " ".join(markdowns),#.replace("\n", " "), 
            model, tokenizer,
        )
        inst = {
            "markdowns": markdowns,
            "topic": topic,  # kind of like the notebook title.  
        } 
        with open(path, "a") as f: f.write(json.dumps(inst)+"\n")
        generated_topics.append(inst)

    return generated_topics

def generate_topics_for_val_data(data: List[dict], model, 
                                 tokenizer, path: str) -> List[dict]:
    generated_topics = []
    pbar = tqdm(data)
    open(path, "w")
    nb_id = 0
    for rec in pbar:
        ctxt = rec["context"][::-1]
        inst_topics = []
        for i, cell in enumerate(ctxt):
            pbar.set_description(f"{i}/{len(ctxt)}")
            if cell["cell_type"] == "markdown":
                proc_md_cell = process_markdown(cell["nl_original"])
                topic = get_bart_tl_topics(
                    proc_md_cell.replace("\n", " "), 
                    model, tokenizer,
                )
                inst = {
                    "nb_id": nb_id, "md_id": i, 
                    "markdown": cell["nl_original"],
                    "topic": topic,
                }
                with open(path, "a") as f: f.write(json.dumps(inst)+"\n")
                inst_topics.append(inst)
        generated_topics.append(inst_topics)
        nb_id += 1

    return generated_topics

# main function.
if __name__ == "__main__":
    input = "site web google search website online internet social content user"
    
    mname = "cristian-popa/bart-tl-ng"
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    # topics = get_bart_tl_topics(input, model, tokenizer)
    # print(f"input: {input}")
    # print(f"topics: {topics}")
    val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
    # path = "./analysis/dev_topics.jsonl"
    # generate_topics_for_val_data(
    #     val_data, model, 
    #     tokenizer, path,
    # )
    path = "./analysis/dev_full_nb_topics.jsonl"
    generate_topics_for_each_nb(
        val_data, model,
        tokenizer, path,
    )