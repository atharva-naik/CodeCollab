# file containing various node type classifiers for different KB sources (as needed).

import os
import json
import torch
import argparse
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers.models.bert.modeling_bert import BertPooler
from transformers import AutoTokenizer, DebertaTokenizerFast, DebertaModel

INIT_POINTS = {"decision tree": "M", "decision tree learning": "M", "machine learning":"C", "artificial intelligence": "C", "epistemology": "C", "cognitive psychology": "C", "psychology": "C", "cognitive linguistics": "C", "digital data": "D", "data": "D", "statistics": "S", "scientific theory": "C", "data structure": "C", "array data structure": "C", "norm": "C", "problem": "T", "informatics": "C", "Computational physiology": "C", "computational science": "C", "hypothesis testing": "S", "algorithm": "M", "ontology": "C", "design": "C", "error": "E", "error detection and correction": "C", "process": "C", "information retrieval": "C", "sampling bias": "C", "cognitive bias": "C", "engineering": "C", "research project": "C", "mathematical model": "M", "partial differential equation": "C", "nonlinear partial differential equation": "C", "conjecture": "C", "poisson bracket": "C", "graph": "C", "method": "M", "generative model": "M", "physics terminology": "C", "area of mathematics": "C", "theory": "C", "discrete mathematics": "C", "logic gate": "M", "polynomial root": "C", "lemma": "C", "computer science": "C", "computer network protocol": "C", "nonparametric regression": "M", "nonparametric statistics": "S", "statistical method": "S", "data scrubbing": "C", "data management": "C", "data extraction": "C", "data processing": "C", "type of test": "E", "modular exponentiation": "M", "integer factorization": "M", "bounded lattice": "C", "maximum": "C", "minimum": "C", "model-free reinforcement learning": "T", "physics": "C", "chemical analysis": "C", "LR parser": "M", "parsing": "T", "field of study": "C", "neuroscience": "C", "applied science": "C", "factorial moment": "S", "Method of moments": "S", "kurtosis": "S", "moment of order r": "S", "moment of order r of a discrete random variable": "S", 'statistic': "S", "correlation coefficient": "S", "Spearman's rank correlation coefficient": "S", "Pearson product-moment correlation coefficient": "S"}
RANDOM_SEED = 2023

# dataset for processing the node names.
class WikiDataNodeDataset(Dataset):
    def __init__(self, data: List[str], labels: List[str]):
        self.data = data
        self.labels = labels
        assert len(labels) == len(data)
        self.label2index = {k: i for i,k in enumerate(json.load(open("./data/DS_KB/semantic_types.json")).keys())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return [self.data[i], self.label2index[self.labels[i]]]

# dataloader class for the node names.
class WikiDataNodeDataLoader(DataLoader):
    def __init__(self, *args, tokenizer=None, **kwargs):
        self.tokenizer = tokenizer
        super(WikiDataNodeDataLoader, self).__init__(
            *args, **kwargs, 
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        enc_inputs = self.tokenizer(
            [item[0] for item in batch], padding="longest", 
            truncation=True, add_special_tokens=True, 
            max_length="50", return_tensors="pt"
        )
        label_tensor = torch.as_tensor([item[1] for item in batch])
        enc_inputs.update({"labels": label_tensor})

        return enc_inputs

def get_args():
    parser = argparse.ArgumentParser(description='Train deberta like classifier on ')
    parser.add_argument('-ts', "--test_size", type=float, default=0.2, 
                        help="size of test data in percent of total data")
    parser.add_argument('-bs', "--batch_size", type=int, default=128,
                        help="batch size used for training")
    parser.add_argument('-d', "--device", type=str, default="cuda:0", help="device used")
    parser.add_argument('-e', "--epochs", type=int, default=10, 
                        help="number of epochs for training")
    parser.add_argument('-es', "eval_steps", type=int, default=200,
                        help="number of steps after which eval is performed")
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-5)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_classes", type=int, default=14)
    args = parser.parse_args()

    return args

# deberta model to classify the semantic type of nodes given the natural language name and the  
class BERTBasedWikiDataNodeClassifier(nn.Module):
    def __init__(self, args, model_path="microsoft/deberta-base", num_classes: int=14):
        self.model_path = model_path
        self.tokenizer = DebertaTokenizerFast.from_pretrained(model_path)
        self.model = DebertaModel.from_pretrained(model_path)
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)
        self.pooler = BertPooler(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def validate(self, test_loader, args):
        test_tot, test_matches = 0
        test_batch_losses = []
        test_bar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="training"
        )
        for step, batch in test_bar:
            for k in batch: batch[k] = batch[k].to(args.device)
            batch_tot = len(batch["labels"])
            test_tot += batch_tot
            with torch.no_grad(): 
                logits, loss = self(**batch)
            # update total matches for train acc.
            batch_matches = (logits.argmax(dim=-1).detach() == batch["labels"]).cpu().sum()
            test_matches += batch_matches
            test_acc = round(100*test_matches/test_tot, 2)
            batch_acc = round(100*batch_matches/batch_tot, 2)
            test_bar.update(f"ba: {batch_acc} a: {test_acc}")

        return test_acc, test_batch_losses, round(np.mean(test_batch_losses), 3)

    def forward(self, labels=None, **args):
        base_model_output = self.model(**args) # base model output (last hidden states)
        pooler_output = self.pooler(base_model_output.last_hidden_state)
        logits = self.classifier(pooler_output)
        if labels is None: return logits, None
        loss = self.loss_fn(logits, labels)
        
        return logits, loss

    def fit(self, data: List[Tuple[str, str]], args):
        X = [x for x,_ in data]
        Y = [y for _,y in data]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=args.test_size, 
            random_state=RANDOM_SEED, stratify=Y
        )

        # move the model to device.
        self.to(args.device)
        train_dataset = WikiDataNodeDataset(X_train, Y_train)
        test_dataset = WikiDataNodeDataset(X_test, Y_test)
        train_loader = WikiDataNodeDataLoader(train_dataset)
        test_loader = WikiDataNodeDataLoader(test_dataset)

        # intialize the optimizer
        self.optimizer = AdamW(
            self.parameters(), 
            lr=args.learning_rate,
        )

        best_acc = 0
        best_epoch = 0
        best_step = 0
        # training loop.
        for epoch_i in range(args.epochs):
            # training step.
            train_tot, train_matches = 0, 0
            train_batch_losses = []
            train_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc="training"
            )
            for step, batch in train_bar:
                for k in batch: batch[k] = batch[k].to(args.device)
                batch_tot = len(batch["labels"])
                train_tot += batch_tot
                logits, loss = self(**batch)
                # compute loss and do optimization
                loss.backward()
                self.optimizer.step()
                # update total matches for train acc.
                batch_matches = (logits.argmax(dim=-1).detach() == batch["labels"]).cpu().sum()
                train_matches += batch_matches
                train_acc = round(100*train_matches/train_tot, 2)
                batch_acc = round(100*batch_matches/batch_tot, 2)
                batch_loss = loss.cpu().item()
                train_batch_losses.append(batch_loss)
                train_bar.update(f"ba: {batch_acc} a: {train_acc} bl: {batch_loss:.3f} l: {np.mean(train_batch_losses):.3f}")
                # do evaluation if eval steps have elapsed.
                if (step+1) % args.eval_steps == 0 or (step+1) == len(train_loader):
                    test_acc, test_batch_losses, test_loss = self.validate(test_loader, args)
                    if best_acc < test_acc:
                        best_step = step
                        best_epoch = (epoch_i+1)
                        best_acc = test_acc
                        print(f"saving best model at: \nstep: {best_step}\nepoch: {best_epoch}\nacc: {best_acc}")
                        checkpoint = {
                            "model_state_dict": self.state_dict(),
                            "optim_state_dict": self.optimizer.state_dict(),
                            "step": best_step, "epoch": best_epoch,
                            "acc": best_acc, "loss": test_loss,
                        }
                        torch.save(checkpoint, "./experiments/DeBERTa_NodeType_Classifier/best_model.pt")

def train_bert_clf():
    """function to fine-tune DeBERTa model on the node to semantic type classification task."""
    # load data.
    data = json.load(open("./data/DS_KB/wikidata_node_classification_data.json"))
    
    # get terminal arguments.
    args = get_args()

    # initialize model and fit it on the data.
    model = BERTBasedWikiDataNodeClassifier(args)
    model.fit(data, args)

# rule based classifier (uses heuristics)
class RuleBasedWikiDataNodeClassifier:
    def __init__(self):
        global INIT_POINTS
        self.known_points = INIT_POINTS
        self.curated_graph = json.load(open("./data/DS_TextBooks/unified_triples.json"))
        self.curated_nodes = {}
        for rec in self.curated_graph:
            self.curated_nodes[rec["sub"][0].lower()] = rec["sub"][1]
            self.curated_nodes[rec["obj"][0].lower()] = rec["obj"][1] 
        # print(self.curated_nodes)
    def override_clf_from_curated_info(self, node_name):
        return self.curated_nodes.get(node_name.lower())

    def __call__(self, node_name: str, adj_list: List[Tuple[str, str]]=[]) -> str:
        children = [x for x,_ in adj_list]
        overridden_class = self.override_clf_from_curated_info(node_name)
        if overridden_class is not None:
            # print(f"overrode class for {node_name}")
            self.known_points[node_name] = overridden_class
            return overridden_class
        if node_name.lower().endswith(" problem"): return "T"
        if node_name.lower().endswith("engineering"): return "C"
        if node_name.lower().endswith(" science"): return "C"
        if node_name.lower().endswith("statistics"): return "S"
        if node_name.lower().endswith(" distribution") or node_name.lower().endswith(" distributions"): # or " distribution " in node_name.lower(): 
            self.known_points[node_name] = "S"
            return "S"
        if "conjecture" in node_name.lower():
            self.known_points[node_name] = "C"
            return "C"
        if node_name.lower().endswith("theory"): "C"
        if node_name.lower().endswith("algorithm"):
            self.known_points[node_name] = "M"
            return "M"
        if node_name in self.known_points: return self.known_points[node_name]
        for name in self.known_points:
            if name in children: return self.known_points[name]
        match_conds = {f"subclass of {name}": class_ for name, class_ in self.known_points.items()}
        match_conds.update({f"instance of {name}": class_ for name, class_ in self.known_points.items()})
        # print(match_conds)
        for Q,P in adj_list:
            Q = Q.strip()
            P = P.strip()
            cond = f"{P} {Q}"
            if cond == "instance of concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "has use statistics":
                self.known_points[node_name] = "S"
                return "S"
            elif cond == "subclass of concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of inequality": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of mathematical concept": 
                self.known_points[node_name] = "C"
                return "C"
            elif cond == "instance of algorithm": 
                self.known_points[node_name] = "M"
                return "M"
            elif cond == "subclass of algorithm": 
                self.known_points[node_name] = "M"
                return "M"
            class_ = match_conds.get(cond, None)
            if class_ is not None: 
                self.known_points[node_name] = class_
                return class_
        if "sampling" in node_name:
            self.known_points[node_name] = "S"
            return "S"
        if node_name.lower().endswith("logy"): 
            self.known_points[node_name] = "C" 
            return "C"

        return "U"

def rule_based_clf_predict():
    wiki_clf = RuleBasedWikiDataNodeClassifier()
    node_to_class_mapping = {}
    wikidata_graph = json.load(open("./data/WikiData/ds_qpq_graph_pruned.json"))
    for Q1, adj_list in tqdm(wikidata_graph.items()):
        adj_list = adj_list["E"]
        node_to_class_mapping[Q1] = wiki_clf(Q1, adj_list)
        for Q2, _ in adj_list:
            if Q2 in node_to_class_mapping: continue
            node_to_class_mapping[Q2] = wiki_clf(Q2, [])
    # successfully classified node percentage.
    unk_count = sum([int(v == 'U') for v in node_to_class_mapping.values()])
    tot = len(node_to_class_mapping)
    succ_count = tot - unk_count
    print(f"{succ_count}/{tot} and {100*(succ_count/tot):.2f}% are successfully classified ({unk_count} are still unknown)")
    with open("./data/DS_KB/wikidata_pred_node_classes.json", "w") as f:
        json.dump(node_to_class_mapping, f, indent=4)
    labeled_data = {k.lower(): v for k,v in node_to_class_mapping.items() if v != "U"}
    labeled_data.update({k.lower(): v for k,v in wiki_clf.curated_nodes.items()})
    # also add PwC data.
    with open("./data/PwC/unified_pwc_triples.json") as f:
        pwc_graph = json.load(f)
        for rec in pwc_graph:
            sub = rec["sub"]
            obj = rec["obj"]
            if sub[0] not in labeled_data: labeled_data[sub[0].lower()] = sub[1] 
            if obj[0] not in labeled_data: labeled_data[obj[0].lower()] = obj[1] 
    with open("./data/DS_KB/wikidata_node_classification_data.json", "w") as f:
        json.dump(list(labeled_data.items()), f, indent=4)

# main
if __name__ == "__main__":
    rule_based_clf_predict()