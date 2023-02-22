import json
import numpy as np
from typing import *
from tqdm import tqdm
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
from datautils.markdown_cell_analysis import process_markdown

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

def extract_keyphrases_for_val_data(data: List[dict], extractor, path: str) -> List[dict]:
    extracted_keyphrases = []
    pbar = tqdm(data)
    open(path, "w")
    nb_id = 0
    for rec in pbar:
        ctxt = rec["context"][::-1]
        inst_extracted = []
        for i, cell in enumerate(ctxt):
            pbar.set_description(f"{i}/{len(ctxt)}")
            if cell["cell_type"] == "markdown":
                proc_md_cell = process_markdown(cell["nl_original"])
                keyphrases = extractor(proc_md_cell.replace("\n", " "))
                inst = {
                    "nb_id": nb_id,
                    "md_id": i, 
                    "markdown": cell["nl_original"],
                    "keyphrases": list(set(kp.lower() for kp in keyphrases.tolist())),
                } # print(keyphrases)
                with open(path, "a") as f: f.write(json.dumps(inst)+"\n")
                inst_extracted.append(inst)
        extracted_keyphrases.append(inst_extracted)
        nb_id += 1

    return extracted_keyphrases

# main
if __name__ == "__main__":
    model_name = "ml6team/keyphrase-extraction-kbir-inspec"
    extractor = KeyphraseExtractionPipeline(model=model_name)
#     text = """
# Keyphrase extraction is a technique in text analysis where you extract the
# important keyphrases from a document. Thanks to these keyphrases humans can
# understand the content of a text very quickly and easily without reading it
# completely. Keyphrase extraction was first done primarily by human annotators,
# who read the text in detail and then wrote down the most important keyphrases.
# The disadvantage is that if you work with a lot of documents, this process
# can take a lot of time. 

# Here is where Artificial Intelligence comes in. Currently, classical machine
# learning methods, that use statistical and linguistic features, are widely used
# for the extraction process. Now with deep learning, it is possible to capture
# the semantic meaning of a text even better than these classical methods.
# Classical methods look at the frequency, occurrence and order of words
# in the text, whereas these neural approaches can capture long-term
# semantic dependencies and context of words in a text.
# """
    # keyphrases = extractor(text.replace("\n", " "))
    # print(keyphrases)
    from datautils import read_jsonl
    val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
    val_keyphrases = extract_keyphrases_for_val_data(
        val_data, extractor=extractor, 
        path="./analysis/dev_keyphrases.jsonl",
    )