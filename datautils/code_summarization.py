# # code to summarize intents of code blocks.
# from datautils import read_jsonl
# from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import json
from typing import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from datautils.code_cell_analysis import process_nb_cell
from transformers import RobertaTokenizer, T5ForConditionalGeneration

def safe_process_nb_cell(code: str) -> str:
    try:
        proc_code = process_nb_cell(code)
    except Exception as e: 
        print(e)
        proc_code = code
    
    return proc_code

class CodeCellSummDataset:
    """Dataset to create code cells,
    create a processed version of it and
    wrap it in a function."""
    def __init__(self, codes: List[str]):
        self.codes = codes

    def __getitem__(self, i: int):
        code = self.codes[i]
        proc_code = safe_process_nb_cell(self.codes[i])
        func_code = "\n".join(["def func():"]+["\t"+d for d in proc_code.split("\n")])

        return {"code": code, "proc_code": proc_code, "func_code": func_code}

    def __len__(self):
        return len(self.codes)

class CodeSummarizer:
    """summarize code to NL intent"""
    def __init__(self, model_path: str="Salesforce/codet5-base-multi-sum"):# "stmnk/codet5-small-code-summarization-python"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def batched_call(self, codes: List[str], max_length: int=40, 
                     skip_special_tokens: bool=True,
                     clean_up_tokenization_spaces: bool=True):
        input_ids = self.tokenizer(codes, return_tensors="pt", 
                                   padding="longest", truncation=True).input_ids.cuda()
        generated_ids = self.model.generate(input_ids, max_length=max_length)

        return self.tokenizer.batch_decode(generated_ids.detach().cpu(), 
               						       skip_special_tokens=skip_special_tokens,
                                           clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def __call__(self, code: str, max_length: int=40, 
                 skip_special_tokens: bool=True):
        input_ids = self.tokenizer(code, return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(input_ids, max_length=max_length)

        return self.tokenizer.decode(generated_ids[0].detach().cpu(), 
               						 skip_special_tokens=skip_special_tokens)
#         # text = "def greet(user): print(f'hello <extra_id_0>!')"
#         input_ids = self.tokenizer(text, return_tensors="pt").input_ids
#         # simply generate a single sequence
#         generated_ids = self.model.generate(input_ids, max_length=max_length)

#         return self.tokenizer.decode(
#             generated_ids[0], 
#             skip_special_tokens=True,
#         )

# # main
# if __name__ == "__main__":
#     code_summarizer = CodeSummarizer()
#     val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
#     code = "def greet(user): print(f'hello <extra_id_0>!')"
#     # val_data[0]["code"]
#     intent = code_summarizer(code)
#     print(code)
#     print(intent)


if __name__ == '__main__':
    # tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    # model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    codes = [
        """for n_neighbors in [1, 5, 10, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(n_neighbors, knn.score(X_test, y_test))
""",
"""config_logging('INFO')
possible = IterChannel(KeyedArtifact(i, DerivedArtifact(expensive_deriver, i, name='expensive')) for i in range(10))
extant = IterChannel(KeyedArtifact(i, ExampleExtantArtifact(i)) for i in keys)
all_ = merge_keyed_channels(possible, extant)
print_chans(all_.tee())
config_logging('WARN')""",
"""m= ipyl.Map(scroll_wheel_zoom=True, center=[-0.3515602939922709, 22.5], 
            zoom=1, layout=ipyw.Layout(width='45%', height='450px'))

dc = ipyl.DrawControl(polygon={'shapeOptions': {'color': 'green', 'weight': 2, 'clickable': True}})
m.add_control(dc)

for centroid in df.geometry:
    m.add_layer(ipyl.GeoJSON(data=mapping(centroid)))""",
"""from ipyleaflet import (
    Map, Marker, TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle,
    CircleMarker, GeoJSON, DrawControl,
    FeatureGroup
)
hi_bbox = {
    "type": "Polygon",
    "coordinates": [
        [
            [
              -171.03515625,
              6.926426847059551
            ],
            [
              -171.03515625,
              33.797408767572485
            ],
            [
              -144.580078125,
              33.797408767572485
            ],
            [
              -144.580078125,
              6.926426847059551
            ],
            [
              -171.03515625,
              6.926426847059551
            ]
        ]
    ]
}
center = [20, -157]
m = Map(center=center, zoom=4)
g = GeoJSON(data=hi_bbox)
m.add_layer(g)
m"""]
    code_summ_path = "./data/juice-dataset/train_KB_batched_summ.jsonl"
    assert not os.path.exists(code_summ_path), "aborting to avoid overwrite issues"
    open(code_summ_path, "w") # create the file
    # codes = [code.strip("\n") for code in codes]
    all_codes = list(json.load(open("./JuICe_train_code_KB.json")).keys())
    csumm_dataset = CodeCellSummDataset(all_codes)
    csumm_dataloader = DataLoader(csumm_dataset, batch_size=32, shuffle=False)
    csumm = CodeSummarizer()
    csumm.model.cuda()
    id = 0
    for batch in tqdm(csumm_dataloader):
        # input_ids = tokenizer(code, return_tensors="pt").input_ids
        # generated_ids = model.generate(input_ids, max_length=20)
        batched_cell_summ = csumm.batched_call(batch["proc_code"])
        batched_func_summ = csumm.batched_call(batch["func_code"])
        for code, proc_code, func_code, cell_summ, func_summ in zip(
            batch["code"], 
            batch["proc_code"], 
            batch["func_code"], 
            batched_cell_summ, 
            batched_func_summ,
        ):
            with open(code_summ_path, "a") as f:
                f.write(json.dumps({
                    "id": id, "code": code,
                    "proc_code": proc_code,
                    "cell_summ": cell_summ,
                    "func_code": func_code,
                    "func_summ": func_summ,
                })+"\n")
            id += 1