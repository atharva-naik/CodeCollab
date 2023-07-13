# extract triples from PwC files:
import ast
import json
from typing import *
from tqdm import tqdm

ALL_NODES = set()
COLLECTIONS = {}
def extract_papers_and_abstracts() -> Dict[str, dict]:
    papers = json.load(open("./data/PwC/papers-with-abstracts.json")) 
    triples = {}
    for rec in tqdm(papers):
        try: sub = rec["title"].strip()
        except: continue # missing title
        # e.g.  12%|█████▏                                      | 45100/378105 [00:00<00:02, 155443.38it/s]{'paper_url': 'https://paperswithcode.com/paper/000000000', 'arxiv_id': '0000.00000', 'title': None, 'abstract': None, 'url_abs': 'https://arxiv.org/abs/0000.00000', 'url_pdf': None, 'proceeding': None, 'authors': [], 'tasks': [], 'date': '', 'methods': []}
        for task in rec["tasks"]:
            obj = task.strip()
            ALL_NODES.add(obj)
            # triples[f"{sub} HAS GOAL(S) {obj}"] = {
            #     "sub": (sub,"C",rec["abstract"]), 
            #     "obj": (obj,"T",""), 
            #     "e": "HAS GOAL(S)"
            # }
            # triples[f"{obj} GOAL OF {sub}"] = {
            #     "sub": (obj,"T",""), 
            #     "obj": (sub,"C",rec["abstract"]), 
            #     "e": "GOAL OF"
            # }
        for method in rec["methods"]:
            obj = method["name"] if method["full_name"] in ["", None] else method["full_name"]
            obj = obj.strip()
            ALL_NODES.add(obj)
            # triples[f"{sub} USES {obj}"] = {
            #     "sub": (sub,"C",rec["abstract"]), 
            #     "obj": (obj,"M",method["description"]), 
            #     "e": "USES"
            # }
            # triples[f"{obj} USED BY {sub}"] = {
            #     "sub": (obj,"M",method["description"]), 
            #     "obj": (sub,"C",rec["abstract"]), 
            #     "e": "USED BY"
            # }
            for task in rec["tasks"]:
                sub1 = task.strip()
                ALL_NODES.add(sub1)
                triples[f"{sub1} MODELED BY {obj}"] = {
                    "sub": (sub1,"T",""), 
                    "obj": (obj,"M",method["description"]), 
                    "e": "MODELED BY"
                }
                triples[f"{obj} CAN MODEL {sub1}"] = {
                    "sub": (obj,"M",method["description"]), 
                    "obj": (sub1,"T",""), 
                    "e": "CAN MODEL"
                }  
            try:
                obj1 = method["main_collection"]["name"].strip()
                desc1 = method["main_collection"]["description"]
                ALL_NODES.add(obj1)
                COLLECTIONS[obj1] = ""
            except: continue
            triples[f"{obj} INSTANCE OF {obj1}"] = {
                "sub": (obj,"M",method["description"]), 
                "obj": (obj1,"M",desc1), 
                "e": "INSTANCE OF"
            }
            triples[f"{obj1} HAS INSTANCE {obj}"] = {
                "sub": (obj1,"M",desc1), 
                "obj": (obj,"M",method["description"]), 
                "e": "HAS INSTANCE"
            } 

    return triples

def process_val_and_delta(metric_str: str, symbol: str="±") -> Tuple[float, float]:
    val, delta = metric_str.split(symbol)
    val = val.replace("%","")
    delta = delta.replace("%","")
    val, delta = resflot(val),resflot(delta)

    return (val, delta)

def resflot(x):
    """assumption: all decimals after the first one should be ignored."""
    # remove commas.
    x = x.replace(",","")
    if "." in x:
        parts = x.split(".")
        return float(f"{parts[0]}.{''.join(parts[1:])}")
    return float(x)

def scalar_metric_convert(metric_str: str) -> Union[Tuple[float, float], float]:
    # hard coded because I'm frustrated:
    metric_str = metric_str.strip("-").strip("–").strip("e")
    if metric_str in ["-", "--", "–", "?", "", "%"]: return None
    if "," in metric_str and not("(" in metric_str or ")" in metric_str) and "." not in metric_str:
        metric_str = metric_str.replace(",",".")  
    if "(" in metric_str:
        metric_str = metric_str.split("(")[0]
    for symbol in ["+/-", "±", "+-", "-+"]:
        if symbol in metric_str: 
            return process_val_and_delta(metric_str, symbol=symbol)
    multiplier = 1
    if metric_str.endswith("M") or metric_str.endswith("m"): 
        metric_str = metric_str[:-1]
        multiplier = 10**6
    elif metric_str.endswith("M+") or metric_str.endswith("m+") or metric_str.endswith("MB"): 
        metric_str = metric_str[:-2]
        multiplier = 10**6   
    elif metric_str.endswith("K") or metric_str.endswith("k"): 
        metric_str = metric_str[:-1]
        multiplier = 10**3
    elif metric_str.endswith("B") or metric_str.endswith("G"): 
        metric_str = metric_str[:-1]
        multiplier = 10**9
    if metric_str == "0.57×1e-4": return 0.57*1e-4

    return multiplier * resflot(metric_str.replace("%","").replace("..",".").replace(":",".").replace("(self-sup.)",""))

ERR_CTR = 0
def extract_evaluation_tables(eval_tables: list=None, pbar=None) -> Dict[str, dict]:
    global ERR_CTR
    if eval_tables is None:
        eval_tables = json.load(open("./data/PwC/evaluation-tables.json"))
        pbar = tqdm(range(1000000))
    triples = {}
    assert isinstance(eval_tables, list), f"{type(eval_tables)} {eval_tables.keys()}"
    for rec in eval_tables:
        sub = rec["task"].strip()
        ALL_NODES.add(sub)
        # link to first level subtasks.
        for subtask in rec["subtasks"]:
            obj = subtask["task"].strip()
            triples[f"{sub} SUPERCLASS OF {obj}"] = {
                "sub": (sub,"T",rec["description"]), 
                "obj": (obj,"T",subtask["description"]), 
                "e": "SUPERCLASS OF"
            }
            triples[f"{obj} SUBCLASS OF {sub}"] = {
                "sub": (obj,"T",subtask["description"]), 
                "obj": (sub,"T",rec["description"]), 
                "e": "SUBCLASS OF"
            }
            pbar.update(2)
            ALL_NODES.add(obj)
        # link to first level datasets:
        for dataset in rec.get("datasets",[]):
            obj = dataset["dataset"].strip()
            ALL_NODES.add(obj)
            triples[f"{sub} GOAL OF {obj}"] = {
                "sub": (sub,"T",rec["description"]), 
                "obj": (obj,"D",dataset["description"]), 
                "e": "GOAL OF"
            }
            triples[f"{obj} HAS GOAL(S) {sub}"] = {
                "sub": (obj,"D",dataset["description"]),
                "obj": (sub,"T",rec["description"]), 
                "e": "HAS GOAL(S)"
            }
            pbar.update(2)
            # link methods in the tables to their scores.
            for row in dataset["sota"]["rows"]:
                sub_ = (row["model_name"], "M", row["paper_title"])
                ALL_NODES.add(sub_[0])
                for metric_name, metric_value in row["metrics"].items():
                    # hard coded out of frustration
                    if metric_value is not None:
                        # take care of removing hard coded strings.
                        for HARD_CODED_STR in ["(non-standard, w/o motion modalities)", "(average of 3 split train/test)", "training label (semi-supervised)", "\ufeff", "mm", "dB", "x", "*", "Please tell us"]:
                            metric_value = metric_value.replace(HARD_CODED_STR,"").strip()
                        for HARD_CODED_STR, HARD_CODED_REPLACEMENT in {"O.753": "0.753", " pm ": "+-"}.items():
                            metric_value = metric_value.replace(HARD_CODED_STR, HARD_CODED_REPLACEMENT)
                    # metric_value based hard-coded stuff.
                    try: 
                        if metric_value is None: pass
                        elif metric_name == "Acc" and dataset["dataset"] == "Stackoverflow" and sub == "Short Text Clustering": pass
                        elif metric_value == "/": pass
                        elif metric_value in ["N/A", "n.a.", "NA"]: metric_value = None
                        elif "SSS: " in metric_value and "NSS: " in metric_value: pass
                        elif "HVLA/HVHA/LVLA/LVHA: " in metric_value: pass
                        # metric name based hard coded handlers for each kind of format.
                        elif metric_name in ["mAP", "AP"]: # mAP/AP metric exception
                            try: metric_value = ast.literal_eval(metric_value.split("on")[0].strip().replace("%",""))
                            except Exception as e: metric_value = scalar_metric_convert(metric_value)
                        elif metric_name == "MAE" and dataset["dataset"] == "deepMTJ v1":
                            metric_value = metric_value.replace("mm","")
                            metric_value = scalar_metric_convert(metric_value)
                        elif metric_name in ["5 fold cross validation", "MLP Hidden Layers-width"]: 
                            metric_value = [scalar_metric_convert(x) for x in metric_value.split("-")]
                        elif metric_name in ["Resolution", "GFLOPs", "FLOPs (G) x views"]: # Resolution exception
                            metric_value = eval(metric_value.replace("x","*"))
                        elif metric_name == "SF-all":
                            metric_value = metric_value.replace(">","").replace("(","").replace(")","")
                            metric_value = scalar_metric_convert(metric_value)
                        elif metric_name == "Mono" and metric_value in ["X","O"]:
                            metric_value = 1 if "X" else 0
                        elif metric_name == "All" and dataset["dataset"] == "custom" and sub == "graph partitioning": pass
                        elif metric_name == "Test Time": 
                            metric_value = resflot(metric_value.replace("s/img",""))
                        elif metric_name in ["MC1", "Validation mIoU", "Test mIoU", 
                                                "Validation Dice Multiclass", "Test Dice Multiclass"]:
                            metric_value = scalar_metric_convert(metric_value.replace(" ",""))
                        elif metric_name in ["@nose", "@mouth", "@forehead", "@cheek"]:
                            metric_value = scalar_metric_convert(metric_value.replace("(","").replace(")",""))
                        elif metric_name in ["1-of-100 Accuracy", "training dataset", "Text model", "Train Split",
                                            "model", "Sentence Encoder", "Cross Sentence", "Train set", "Training Split",
                                            "Type", "Multi-View or Monocular", "2D detector", "Backbone",
                                            "3D Annotations", "14 gestures accuracy", "L2 Norm", "Train Set",
                                            "Notes", "Category", "FPS on CPU", "Network", "Architecture",
                                            "10 way 5~10 shot", "Pretraining Dataset", "PRE-TRAINING DATASET",
                                            "Pretrain", "Pre-Training Dataset", "Actions Top-1 (S2)", "PSNR",
                                            "Average PSNR", "detector"]: 
                            pass # confusing metrics: left as it is.
                        elif metric_name in ["Interpretable", "Source-free", "Multi-Task Supervision", "Pretrained",
                                             "Source-Free", "Single-view", "Unsupervised", "10 fold Cross validation",
                                             "Using 2D ground-truth joints", "Frozen", "official split", "Validation",
                                             "Use Video Sequence", "Need Ground Truth 2D Pose", "ImageNet Pretrained",
                                             "Pretrained/Transfer Learning", "Ext. data"]: # boolean metrics
                            try: 
                                metric_value = {
                                    "yes": 1, "no": 0, "y": 1, "n": 0, 
                                    "false": 0, "true": 1
                                }[metric_value.lower()]
                            except Exception as e: # weird exception for "Human3.6M", "Human Pose Estimation" task.
                                metric_value = 1
                        elif "/" in metric_value: #["LH/RH-MVE", "LH/RH-MPJPE"]:
                            metric_value = [resflot(metric_val) for metric_val in metric_value.split("/")]
                        elif "|" in metric_value:
                            metric_value = [resflot(metric_val) for metric_val in metric_value.split("|")]

                        # default handling case,
                        else: metric_value = scalar_metric_convert(metric_value)
                    except Exception as e: 
                        # print(e)
                        # print(f'\x1b[31;1mERROR\x1b[0m\n\ttask: {sub}\n\tdataset: {dataset["dataset"]}\n\tmetric_name: {metric_name}\n\tmetric value: {metric_value}')
                        # exit()
                        ERR_CTR += 1
                        pbar.set_description(f"{ERR_CTR} errors")
                    obj_ = (metric_name, "E", "")
                    ALL_NODES.add(obj_[0])
                    triples[f"{sub_[0]} HAS SCORE {obj_[0]}"] = {
                        "sub": sub_, "obj": obj_, 
                        "e": "HAS SCORE", "w": metric_value,
                    }
                    triples[f"{obj_[0]} IS SCORE OF {sub_[0]}"] = {
                        "sub": obj_, "obj": sub_, 
                        "e": "IS SCORE OF", "w": metric_value
                    }
                    pbar.update(2)
            # link to first level metrics.
            for metric in dataset["sota"]["metrics"]:
                obj = metric.strip()
                ALL_NODES.add(obj)
                triples[f"{sub} EVALUATED BY {obj}"] = {
                    "sub": (sub,"T",rec["description"]), 
                    "obj": (obj,"E",""), 
                    "e": "EVALUATED BY"
                }
                triples[f"{obj} EVALUATES {sub}"] = {
                    "sub": (obj,"E",""), 
                    "obj": (sub,"T",rec["description"]), 
                    "e": "EVALUATES"
                }   
                triples[f"{sub} EVALUATED BY {obj}"] = {
                    "sub": (dataset["dataset"].strip(),"D",dataset["description"]), 
                    "obj": (obj,"E",""), 
                    "e": "EVALUATED BY"
                }
                triples[f"{obj} EVALUATES {sub}"] = {
                    "sub": (dataset["dataset"].strip(),"E",""), 
                    "obj": (sub,"D",dataset["description"]), 
                    "e": "EVALUATES"
                }
                pbar.update(4)
        # link to first level subtasks.
        triples.update(extract_evaluation_tables(eval_tables=rec["subtasks"], pbar=pbar))

    return triples

def extract_methods() -> Dict[str, dict]:
    """extract triples from the `methods.json` file"""
    methods = json.load(open("./data/PwC/methods.json"))
    triples = {}
    for rec in methods:
        sub = rec["name"] if rec["full_name"] in ["", None] else rec["full_name"]
        ALL_NODES.add(sub)
        sub = sub.strip()
        try: obj = rec["paper"]["title"].strip()
        except: obj = rec["source_title"]
        # links between the paper and dataset.
        if obj is not None: pass
            # triples[f"{sub} USED BY {obj}"] = {
            #     "sub": (sub,"M",rec["description"]), 
            #     "obj": (obj,"C",""), 
            #     "e": "USED BY"
            # }
            # triples[f"{obj} USES {sub}"] = {
            #     "sub": (obj,"C",""), 
            #     "obj": (sub,"M",rec["description"]), 
            #     "e": "USES"
            # }
        for col in rec["collections"]:
            obj = col["collection"].strip()
            ALL_NODES.add(obj)
            COLLECTIONS[obj] = ""
            triples[f"{sub} INSTANCE OF {obj}"] = {
                "sub": (sub,"M",rec["description"]), 
                "obj": (obj,"M",""), 
                "e": "INSTANCE OF"
            }
            triples[f"{obj} HAS INSTANCE {sub}"] = {
                "sub": (obj,"M",""), 
                "obj": (sub,"M",rec["description"]), 
                "e": "HAS INSTANCE"
            }

    return triples    

def extract_datasets() -> Dict[str, dict]:
    """extract triples from the `datasets.json` file"""
    datasets = json.load(open("./data/PwC/datasets.json"))
    triples = {}
    for rec in datasets:
        sub = rec["name"] if rec["full_name"] in ["", None] else rec["full_name"]
        sub = sub.strip()
        ALL_NODES.add(sub)
        try: obj = rec["paper"]["title"].strip()
        except: obj = None
        # links between the paper and dataset.
        if obj is not None: pass
            # triples[f"{sub} USED BY {obj}"] = {
            #     "sub": (sub,"D",rec["description"]), 
            #     "obj": (obj,"C",""), 
            #     "e": "USED BY"
            # }
            # triples[f"{obj} USES {sub}"] = {
            #     "sub": (obj,"C",""), 
            #     "obj": (sub,"D",rec["description"]), 
            #     "e": "USES"
            # }
        for task in rec["tasks"]:
            obj = task["task"].strip()
            ALL_NODES.add(obj)
            # links between task and dataset.
            triples[f"{sub} USED BY {obj}"] = {
                "sub": (sub,"D",rec["description"]), 
                "obj": (obj,"T",""), 
                "e": "USED BY"
            }
            triples[f"{obj} USES {sub}"] = {
                "sub": (obj,"T",""), 
                "obj": (sub,"D",rec["description"]), 
                "e": "USES"
            }  
        for modality in rec["modalities"]:
            obj = modality.strip()
            ALL_NODES.add(obj)
            # links between modality and dataset.
            triples[f"{sub} HAS MODALITY {obj}"] = {
                "sub": (sub,"D",rec["description"]), 
                "obj": (obj,"m",""), 
                "e": "HAS MODALITY"
            }
            triples[f"{obj} HAS DATASET {sub}"] = {
                "sub": (obj,"m",""), 
                "obj": (sub,"D",rec["description"]), 
                "e": "HAS DATASET"
            }  
        for lang in rec["languages"]:
            obj = lang.strip()
            ALL_NODES.add(obj)
            # links between language and dataset.
            triples[f"{sub} HAS LANGUAGE {obj}"] = {
                "sub": (sub,"D",rec["description"]), 
                "obj": (obj,"l",""), 
                "e": "HAS LANGUAGE"
            }
            triples[f"{obj} HAS DATASET {sub}"] = {
                "sub": (obj,"l",""), 
                "obj": (sub,"D",rec["description"]), 
                "e": "HAS DATASET"
            }            
        for variant in rec["variants"]:
            obj = variant.strip()
            ALL_NODES.add(obj)
            if obj == sub: continue
            # links between the paper and dataset.
            triples[f"{sub} HAS VARIANT {obj}"] = {
                "sub": (sub,"D",rec["description"]), 
                "obj": (obj,"D",""), 
                "e": "HAS VARIANT"
            }
            triples[f"{obj} HAS VARIANT {sub}"] = {
                "sub": (obj,"D",""), 
                "obj": (sub,"D",rec["description"]), 
                "e": "HAS VARIANT"
            }

    return triples

# main
if __name__ == "__main__":
    all_triples = {}
    # datasets:
    datasets_triples = extract_datasets()
    print(f"datasets: {len(datasets_triples)}")
    all_triples.update(datasets_triples)
    # methods:
    methods_triples = extract_methods()
    print(f"methods: {len(methods_triples)}")
    all_triples.update(methods_triples)
    # evaluation tables:
    eval_triples = extract_evaluation_tables()
    print(f"evaluation tables: {len(eval_triples)}")
    all_triples.update(eval_triples)
    # papers and abstracts:
    paper_triples = extract_papers_and_abstracts()
    print(f"paper tables: {len(paper_triples)}")
    all_triples.update(paper_triples)
    # print length of unified dataset.
    print(len(datasets_triples)+len(methods_triples)+len(eval_triples)+len(paper_triples))
    print("|E|:", len(all_triples))
    print("|N|:", len(ALL_NODES))
    with open("./data/PwC/unified_method_collections.json", "w") as f:
        json.dump(list(COLLECTIONS.keys()), f, indent=4)
    with open("./data/PwC/unified_pwc_triples.json", "w") as f:
        json.dump(list(all_triples.values()), f, indent=4)