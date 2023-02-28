#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code for training/fine-tuning CodePLMs for code search using bi-encoder setups.
import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

# import the dataset classes needed for code search for various datasets.
from datautils.dataloaders import CoNaLaCodeBERTCodeSearchDataset

# bi-encoder for code search.
class CodeBERTSearchModel(nn.Module):
    """
    This class implements the classic bi-encoder architecture
    for code search using two CodePLM instances.
    """
    def __init__(self, path: str="microsoft/codebert-base", device: str="cuda"):
        super(CodeBERTSearchModel, self).__init__()
        self.code_encoder = RobertaModel.from_pretrained(path)
        self.query_encoder = RobertaModel.from_pretrained(path)
        self.ce_loss = nn.CrossEntropyLoss()
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"moving model to device: {self.device}")
        self.to(device)

    def forward(self, query_iids, query_attn, code_iids, code_attn):
        query_enc = self.query_encoder(query_iids, query_attn).pooler_output
        code_enc = self.code_encoder(code_iids, code_attn).pooler_output
        batch_size, _ = query_enc.shape
        target = torch.as_tensor(range(batch_size)).to(self.device)
        scores = query_enc @ code_enc.T
        loss = self.ce_loss(scores, target)

        return query_enc, code_enc, query_enc @ code_enc.T, loss

def codebert_codesearch_val(model, dataloader, log_file, args):
    model.eval()
    tot, matches, batch_losses = 0, 0, []
    pbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader),
    )
    for step, batch in pbar:
        model.zero_grad()
        with torch.no_grad():
            for j in range(len(batch)):
                batch[j] = batch[j].to(args.device)
            _, _, scores, loss = model(*batch)
            batch_losses.append(loss.detach().cpu().item())
            preds = scores.cpu().argmax(dim=-1)
            # compute train classification accuracy for code search.
            batch_tot = len(batch[0])
            batch_matches = (preds == torch.as_tensor(range(batch_tot))).sum().item()
            # epoch level accuracy.
            tot += batch_tot
            matches += batch_matches
            pbar.set_description(
                f"V: bl: {batch_losses[-1]:.3f} l: {np.mean(batch_losses):.3f} ba: {(100*batch_matches/batch_tot):.2f} a: {(100*matches/tot):.2f}"
            )
            if step%args.log_steps == 0 or (step+1) == len(dataloader): 
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "bl": batch_losses[-1],
                        "l": np.mean(batch_losses),
                        "ba": 100*batch_matches/batch_tot,
                        "a": 100*matches/tot,
                    })+"\n")

    return matches/tot

def print_args(args):
    for k,v in vars(args).items():
        print(f"{k}: {v}")

def codebert_codesearch_finetune(args):
    device = args.device
    codesearch_biencoder = CodeBERTSearchModel(
        args.model_path, device=device,
    )
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    tok_args = {
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": True,
        "max_length": 100,
    }
    config = vars(args)
    config["tokenizer_args"] = tok_args
    
    trainset = CoNaLaCodeBERTCodeSearchDataset(split="train", tokenizer=tokenizer, **tok_args)
    valset = CoNaLaCodeBERTCodeSearchDataset(split="val", tokenizer=tokenizer, **tok_args)
    testset = CoNaLaCodeBERTCodeSearchDataset(split="test", tokenizer=tokenizer, **tok_args)
    optimizer = AdamW(
        codesearch_biencoder.parameters(),
        lr=args.learning_rate,
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    exp_folder = os.path.join(
        "experiments", 
        args.experiment_name,
    )
    
    # create new folder for the experiment.
    os.makedirs(exp_folder, exist_ok=True)
    config_path = os.path.join(exp_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print_args(args)

    pbar = tqdm(
        enumerate(trainloader),
        total=len(trainloader),
    )
    # create directory for logging stats.
    best_val_acc = 0
    logs_dir = os.path.join(exp_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    for epoch in range(args.epochs):
        train_log_file = os.path.join(logs_dir, f"train_epoch_{epoch+1}.jsonl")
        tot, matches, batch_losses = 0, 0, []
        for step, batch in pbar:
            codesearch_biencoder.train()
            codesearch_biencoder.zero_grad()
            for j in range(len(batch)):
                batch[j] = batch[j].to(device)
            _, _, scores, loss = codesearch_biencoder(*batch)
            # model update.
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach().cpu().item())
            preds = scores.cpu().argmax(dim=-1)
            # compute train classification accuracy for code search.
            batch_tot = len(batch[0])
            batch_matches = (preds == torch.as_tensor(range(batch_tot))).sum().item()
            # print(batch_tot, batch_matches)
            # epoch level accuracy.
            tot += batch_tot
            matches += batch_matches
            pbar.set_description(
                f"T: {epoch+1}/{args.epochs}: bl: {batch_losses[-1]:.3f} l: {np.mean(batch_losses):.3f} ba: {(100*batch_matches/batch_tot):.2f} a: {(100*matches/tot):.2f}"
            )
            if step%args.log_steps == 0 or (step+1) == len(trainloader): 
                with open(train_log_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "bl": batch_losses[-1],
                        "l": np.mean(batch_losses),
                        "ba": 100*batch_matches/batch_tot,
                        "a": 100*matches/tot,
                    })+"\n")
            if (step+1)%args.eval_steps == 0 or (step+1) == len(trainloader):
                val_log_file = os.path.join(logs_dir, f"val_epoch_{epoch+1}_step_{step}.jsonl")
                val_acc = codebert_codesearch_val(codesearch_biencoder, valloader, val_log_file, args)
                # save the best model.
                if val_acc > best_val_acc:
                    save_dict = {
                        "step": step, 
                        "epoch": epoch+1,
                        "state_dict": codesearch_biencoder.state_dict(),
                        "val_acc": best_val_acc,
                    }
                    model_save_path = os.path.join(
                        exp_folder, "best_model.pt"
                    )
                    torch.save(save_dict, model_save_path)
    test_log_file = os.path.join(exp_folder, f"test_log.jsonl")
    codebert_codesearch_val(codesearch_biencoder, testloader, test_log_file, args)

# get training/validation arguments.
def get_args():
    parser = argparse.ArgumentParser(
        description='''script to finetune CodeBERT for 
code search using a bi-encoder setup''')
    parser.add_argument('-d', '--device', default="cuda", type=str, 
                        help="device to be used for fine-tuning")
    parser.add_argument('-bs', '--batch_size', default=32, 
                        type=int, help="batch size used for training")
    parser.add_argument('-mp', "--model_path", type=str,
                        default="microsoft/codebert-base",
                        help="path to model/hf model name")
    parser.add_argument("-lr", "--learning_rate", type=float,
                        default=1e-5, help="learning rate for training")
    parser.add_argument("-exp", "--experiment_name", required=True, 
                        help="name of the experiment")
    parser.add_argument("-e", "--epochs", type=int, default=5, 
                        help="number of epochs for training")
    parser.add_argument("-ls", "--log_steps", type=int, default=20,
                        help="log training stats after these many steps")
    parser.add_argument("-es", "--eval_steps", type=int, default=500,
                        help="do validation after these many steps")
    args = parser.parse_args()

    return args

# main
if __name__ == "__main__":
    args = get_args()
    codebert_codesearch_finetune(args)