import os
import json
import time
import torch
import random
import argparse
import numpy as np
from typing import *
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModel
from datautils.dataloaders import SimpleCellSeqDataLoader, SimpleCellSeqDataset, InCoderCellSeqDataset, DataLoader

# seed using random, numpy, torch.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_args():
    parser = argparse.ArgumentParser("""script to train simple cell sequence prediction language models""")
    parser.add_argument("-tp", "--train_path", 
                        default="./data/juice-dataset/train.jsonl", 
                        type=str, help="path to train dataset")
    parser.add_argument("-vp", "--val_path", 
                        default="./data/juice-dataset/dev.jsonl", 
                        type=str, help="path to validation dataset")
    parser.add_argument("-exp", "--exp", required=True, type=str, 
                        help="name of the experiment folder")
    parser.add_argument("-lr", "--lr", default=1e-5, 
                        type=float, help="learning rate")
    parser.add_argument("-ls", "--log_steps", default=2000, 
                        type=int, help="log steps for val")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, 
                        help="batch size for training and validation")
    parser.add_argument("-e", "--num_epochs", type=int, default=5, 
                        help="number of epochs for training")
    parser.add_argument("-as", "--accum_steps", type=int, default=8,
                        help="number of gradient accumulation steps")

    args = parser.parse_args()

    return vars(args)

# model for predicting the sequence of cell types.
class LSTMCellSeqLM(nn.Module):
    """simple LSTM based Seq2Seq/LM model
    reference: https://github.com/pytorch/examples/blob/main/word_language_model/model.py"""
    def __init__(self, emb_dim: int=100, hidden_dim: int=50,
                 dropout_rate: float=0.1, num_layers: int=1):
        super(LSTMCellSeqLM, self).__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.cell_type_map = {
            "start": 0, "raw": 1, "code": 2,
            "markdown": 3, "heading": 4, "end": 5,
        }
        self.vocab_size = len(self.cell_type_map)
        self.padding_idx = self.vocab_size #len(self.cell_type_map)
        self.encoder = nn.Embedding(
            self.vocab_size+1, emb_dim, 
            padding_idx=self.padding_idx,
        )
        self.lstm_lm = nn.LSTM(
            emb_dim, hidden_dim, # batch_first=True,
            dropout=dropout_rate, num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(
            hidden_dim, 
            len(self.cell_type_map),
        )
        self.init_weights()

    def get_config(self) -> dict:
        return {
            "emb_dim": self.emb_dim,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "padding_idx": self.padding_idx,
            "dropout_rate": self.dropout_rate,
        }

    def init_hidden(self, batch_size: int=64):
        """initialize hidden states of the LSTM"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        """given a batch of cell seq types (with padding), 
        and masks containing padding information, predict
        cell seq. types (teacher forcing type)
        ## Input
        - x: sequece of token types (batch_size).
        - hidden: (initial hidden state, initial cell state) (batch_size x hidden_dim).
        """
        # embeds: batch_size x emb_dim
        embeds = self.dropout(self.encoder(x))
        # lstm output (x) has batch_size x hidden_dim.        
        x, hidden = self.lstm_lm(embeds, hidden) 
        # apply a dropout layer over LSTM output
        x = self.dropout(x)
        # decoder output.
        x = self.decoder(x)
        # obtain log probabilities.
        x.view(-1, self.vocab_size)
        # x = F.log_softmax(x.view(-1, self.vocab_size), dim=-1)
        return x, hidden

# model for predicting the sequence of cell types given the sequence of cell types
# and the previous cells and their contents.
class CodeBERTLSTMCellSeqLM(nn.Module):
    """simple CodeBERT+LSTM based Seq2Seq/LM model
    reference: https://github.com/pytorch/examples/blob/main/word_language_model/model.py"""
    def __init__(self, bert_name: str="microsoft/codebert-base", 
                 type_emb_dim: int=256, hidden_dim: int=512, 
                 dropout_rate: float=0.1, num_layers: int=1):
        super(CodeBERTLSTMCellSeqLM, self).__init__()
        if bert_name.endswith("-base"):
            self.emb_dim = 768
        self.type_emb_dim = type_emb_dim
        self.bert_name = bert_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.cell_type_map = {
            "start": 0, "raw": 1, "code": 2,
            "markdown": 3, "heading": 4, "end": 5,
        }
        self.vocab_size = len(self.cell_type_map)
        self.padding_idx = self.vocab_size #len(self.cell_type_map)
        self.encoder = AutoModel.from_pretrained(bert_name)
        self.cell_type_encoder = nn.Embedding(
            self.vocab_size+1, type_emb_dim, 
            padding_idx=self.padding_idx,
        )
        self.lstm_lm = nn.LSTM(
            self.emb_dim, hidden_dim, # batch_first=True,
            dropout=dropout_rate, num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder = nn.Linear(
            hidden_dim, 
            len(self.cell_type_map),
        )
        self.init_weights()

    def get_config(self) -> dict:
        return {
            "emb_dim": self.emb_dim,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "padding_idx": self.padding_idx,
            "dropout_rate": self.dropout_rate,
            "type_emb_dim": self.type_emb_dim,
        }

    def init_hidden(self, batch_size: int=64):
        """initialize hidden states of the LSTM"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, 
                batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, 
                batch_size, self.hidden_dim))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(
            self.cell_type_encoder.weight, 
            -initrange, initrange
        )
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, content_ids, content_mask, 
                cell_types, hidden_state):
        """given a batch of cell seq types (with padding), 
        and masks containing padding information, predict
        cell seq. types (teacher forcing type)
        ## Input
        - x: sequece of token types (batch_size).
        - hidden: (initial hidden state, initial cell state) (batch_size x hidden_dim).
        """
        # content embedding: batch_size x emb_dim
        content_emb = self.encoder(content_ids, content_mask).pooler_output
        # cell type embedding: batch_size x type_emb_dim
        cell_type_emb = self.cell_type_encoder(cell_types)
        # input embedding: batch_size x (emb_dim + type_emb_dim)
        input_emb = self.dropout(torch.cat((content_emb, 
                                 cell_type_emb), dim=-1))
        # lstm output (x) has batch_size x hidden_dim.        
        x, hidden = self.dropout(self.lstm_lm(input_emb, hidden_state))
        # decoder output.
        x = self.decoder(x)
        # obtain log probabilities.
        x.view(-1, self.vocab_size)
        # x = F.log_softmax(x.view(-1, self.vocab_size), dim=-1)
        return x, hidden

# InCoder model based context to next cell type prediction:
class InCoderCellSeqClf(nn.Module):
    def __init__(self, model_name: str="facebook/incoder-1B", 
                 hidden_dim: int=2048, num_classes: int=4,
                 dropout_rate: float=0.2):
        super(InCoderCellSeqClf, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.incoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.num_classes = num_classes
        self.clf = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss() 

    def load_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(ckpt['state_dict'])

    def get_config(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "model_name": self.model_name,
            # "vocab_size": self.vocab_size,
            # "hidden_dim": self.hidden_dim,
            # "num_layers": self.num_layers,
            # "padding_idx": self.padding_idx,
            "dropout_rate": self.dropout_rate,
        }

    def forward(self, ctxt_ids, ctxt_mask, labels=None):
        loss = None
        hidden_states = self.incoder(
            ctxt_ids, ctxt_mask, 
            output_hidden_states=True,
        ).hidden_states[-2] # use second last hidden state
        incoder_emb = self.dropout(hidden_states.mean(axis=1))
        # mean pool the hidden states across the sequence dim.
        logits = self.clf(incoder_emb)
        # compute loss if labels are provided.
        if labels is not None: 
            loss = self.loss_fn(logits, labels)
        
        return logits, loss

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor): return h.detach()
    else: return tuple(repackage_hidden(v) for v in h)

# def validate_lm(model, dataloader, loss_fn, **args):
#     device = args.get("device", "cuda:0")
#     pbar = tqdm(enumerate(dataloader),
#                 total=len(dataloader))
#     matches, tot = 0, 0
#     val_seq_losses = []
#     for step, batch in pbar:
#         N, S = batch[0].shape
#         gold_input = batch[0].transpose(1,0)
#         target = batch[1].transpose(1,0) # batch_size x seq_len -> seq_len x batch_size
#         # x = torch.full((1, N), 0) # input for teacher forcing, 0 is the start token.
#         hidden = model.init_hidden(N)
#         for i in range(len(hidden)):
#             hidden[i].to(device)
#         seq_losses = []
#         for ind in range(S-1):
#             model.eval()
#             with torch.no_grad():
#                 x = gold_input[ind].unsqueeze(dim=0).to(device)
#                 hidden = repackage_hidden(hidden)
#                 y, hidden = model(x, hidden)
#                 # try to predict next token.
#                 y_true = target[ind+1].to(device)
#                 batch_loss = loss_fn(y.squeeze(), y_true)
#                 # obtain predictions and move them to cpu.
#                 preds = y.squeeze().argmax(dim=-1).cpu()
#                 matches += (preds == y_true.cpu()).sum().item()
#                 tot += N
#                 seq_losses.append(batch_loss.item())
#                 pbar.set_description(f"V: l: {np.mean(val_seq_losses):.3f} bl: {batch_loss.item():.3f} sl: {np.mean(seq_losses):.3f} a: {(100*matches/tot):.2f}")
#                 model.zero_grad()
#         val_seq_losses.append(np.mean(seq_losses))

#     return matches/tot, np.mean(val_seq_losses), val_seq_losses

# def train_lm(**args):
#     device = args.get("device", "cuda:0")
#     val_path = args["val_path"]
#     log_steps = args["log_steps"]
#     train_path = args["train_path"]
#     batch_size = args["batch_size"]
#     num_epochs = args["num_epochs"]
#     exp_folder = args["exp"]
#     # create experiment folder:
#     os.makedirs(exp_folder, exist_ok=True)
#     model_save_path = os.path.join(exp_folder, "model.ckpt")
#     config_save_path = os.path.join(exp_folder, "config.json")

#     train_dataset = SimpleCellSeqDataset(train_path) 
#     train_dataloader = SimpleCellSeqDataLoader(
#         train_dataset, shuffle=True,
#         batch_size=batch_size,
#     )
#     # move model to device.
#     model = LSTMCellSeqLM()
#     model.to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     # save config file.
#     config = {}
#     config.update(args)
#     config["model"] = model.get_config()
#     config["loss_fn"] = loss_fn
#     with open(config_save_path, "w") as f:
#         json.dump(args, f, indent=4)

#     val_dataset = SimpleCellSeqDataset(val_path)
#     val_dataloader = SimpleCellSeqDataLoader(
#         val_dataset, shuffle=False,
#         batch_size=batch_size,
#     )
#     lr = args["lr"]
#     optimizer = AdamW(
#         model.clf.parameters(), 
#         eps=1e-8, lr=lr,
#     )
    
#     stats = []
#     best_val_loss = 1000
#     best_epoch_and_step = (0, 0)
#     for epoch in range(num_epochs):
#         stats.append({
#             "batch_seq_losses": [],
#             "train_matches": 0,
#             "train_tot": 0,
#         })
#         pbar = tqdm(enumerate(train_dataloader),
#                     total=len(train_dataloader))
#         model.train()
#         for step, batch in pbar:
#             N, S = batch[0].shape
#             # batch_size x seq_len -> seq_len x batch_size.
#             gold_input = batch[0].transpose(1,0)
#             target = batch[1].transpose(1,0)
#             # initial hidden state for the LSTM (all zeros).
#             hidden = model.init_hidden(N)
#             # hidden state is a tuple.
#             for i in range(len(hidden)):
#                 hidden[i].to(device)
#             seq_losses = []
#             for ind in range(S-1):
#                 model.train()
#                 # input for current stage (for teacher forcing.)
#                 x = gold_input[ind].unsqueeze(dim=0).to(device)
#                 hidden = repackage_hidden(hidden)
#                 y, hidden = model(x, hidden)
#                 # try to predict next token.
#                 y_true = target[ind+1].to(device)
#                 batch_loss = loss_fn(y.squeeze(), y_true)
#                 batch_loss.backward()
#                 optimizer.step()
#                 model.zero_grad()
#                 # obtain predictions and move them to cpu.
#                 preds = y.squeeze().argmax(dim=-1).cpu()
#                 stats[-1]["train_matches"] += (preds == y_true.cpu()).sum().item()
#                 stats[-1]["train_tot"] += N
#                 seq_losses.append(batch_loss.item())
#                 pbar.set_description(f"T: l: {np.mean(stats[-1]['batch_seq_losses']):.3f} bl: {batch_loss.item():.3f} sl: {np.mean(seq_losses):.3f} a: {(100*stats[-1]['train_matches']/stats[-1]['train_tot']):.2f}")
#             stats[-1]["batch_seq_losses"].append(np.mean(seq_losses))
#             # validation loss
#             if ((step+1) % log_steps == 0) or ((step+1) == len(pbar)):
#                 val_acc, val_loss, val_seq_losses = validate_lm(model, val_dataloader, loss_fn, **args)
#                 stats[-1]["val_seq_losses"] = val_seq_losses
#                 if best_val_loss > val_loss:
#                     best_val_loss = val_loss
#                     best_epoch_and_step = (epoch, step)
#                     save_dict = {
#                         "val_acc": val_acc,
#                         "state_dict": model.state_dict(),
#                         "best_val_loss": best_val_loss,
#                         "best_epoch_and_step": best_epoch_and_step,
#                     }
#                     stats[-1]["best_epoch_and_step"] = best_epoch_and_step
#                     stats[-1]["best_val_loss"] = best_val_loss # best validation loss.
#                     stats[-1]["val_acc"] = val_acc # ultimate validation accuracy.
#                     print(f"\n\nsaving \x1b[32;1mbest model\x1b[0m with loss of {best_val_loss:.3f}\n\n")
#                     torch.save(save_dict, model_save_path)
#                 # update saved stats.
#                 stats_save_path = os.path.join(exp_folder, "stats.json")
#                 with open(stats_save_path, "w") as f:
#                     json.dump(stats, f, indent=4)
def test_cell_type_clf(**args):
    device = args.get("device", "cuda:0")
    train_path = args["test_path"]
    batch_size = args["batch_size"]
    exp_folder = args["exp"]
    # load model and move it GPU.
    model_save_path = os.path.join(exp_folder, "model.ckpt")
    model = InCoderCellSeqClf()
    model.load_checkpoint(model_save_path)
    model.to(device)

    matches, tot = 0, 0
    val_batch_losses = []

    for step, batch in pbar:
        for i in range(len(batch)): # move tensors to GPU.
            batch[i] = batch[i].to(device)
        model.train() # set up for training.
        with torch.no_grad():
            logits, loss = model(batch[0], batch[1], labels=batch[2])
            model.zero_grad()
            preds = logits.argmax(dim=-1).cpu()
            matches += (preds == batch[2].cpu()).sum().item()
            tot += len(batch[0])
        pbar.set_description(f"V: l: {np.mean(val_batch_losses):.3f} bl: {loss.item():.3f} sl: {np.mean(val_batch_losses):.3f} a: {(100*matches/tot):.2f}")
        val_batch_losses.append(loss.item())

    return matches/tot, val_batch_losses

def validate_cell_type_clf(model, dataloader, **args):
    device = args.get("device", "cuda:0")
    pbar = tqdm(enumerate(dataloader),
                total=len(dataloader))
    matches, tot = 0, 0
    val_batch_losses = []
    model.eval()
    for step, batch in pbar:
        for i in range(len(batch)): # move tensors to GPU.
            batch[i] = batch[i].to(device)
        model.train() # set up for training.
        with torch.no_grad():
            logits, loss = model(batch[0], batch[1], labels=batch[2])
            model.zero_grad()
            preds = logits.argmax(dim=-1).cpu()
            matches += (preds == batch[2].cpu()).sum().item()
            tot += len(batch[0])
        pbar.set_description(f"V: l: {np.mean(val_batch_losses):.3f} bl: {loss.item():.3f} sl: {np.mean(val_batch_losses):.3f} a: {(100*matches/tot):.2f}")
        val_batch_losses.append(loss.item())

    return matches/tot, val_batch_losses

def train_cell_type_clf(**args):
    print(args)
    device = args.get("device", "cuda:0")
    val_path = args["val_path"]
    log_steps = args["log_steps"]
    train_path = args["train_path"]
    batch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    accum_steps = args["accum_steps"]
    exp_folder = args["exp"]
    # create experiment folder:
    os.makedirs(exp_folder, exist_ok=True)
    model_save_path = os.path.join(exp_folder, "model.ckpt")
    config_save_path = os.path.join(exp_folder, "config.json")

    train_dataset = InCoderCellSeqDataset(train_path) 
    train_dataloader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=batch_size,
    )
    val_dataset = InCoderCellSeqDataset(val_path)
    val_dataloader = DataLoader(
        val_dataset, shuffle=False,
        batch_size=batch_size,
    )

    # move model to device.
    s = time.time()
    print("loading incoder-based classifier")
    model = InCoderCellSeqClf()
    model.to(device)
    print(f"loaded incoder model in {time.time()-s}s")
    # save config file.
    config = {}
    config.update(args)
    config["model"] = model.get_config()
    config["loss_fn"] = model.loss_fn
    with open(config_save_path, "w") as f:
        json.dump(args, f, indent=4)

    lr = args["lr"]
    optimizer = AdamW(
        model.parameters(), 
        eps=1e-8, lr=lr,
    )
    
    stats = []
    best_val_acc = 0
    best_epoch_and_step = (0, 0)
    for epoch in range(num_epochs):
        stats.append({
            "batch_losses": [],
            "train_matches": 0,
            "train_tot": 0,
        })
        pbar = tqdm(enumerate(train_dataloader),
                    total=len(train_dataloader))
        for step, batch in pbar:
            input_ids = batch[0].cuda()
            attn_mask = batch[1].cuda()
            labels = batch[2].cuda()
            model.train() # set up for training.
            logits, loss = model(
                input_ids, attn_mask, 
                labels=labels,
            )
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            if ((step+1)%accum_steps == 0) or (step+1 == len(pbar)):
                optimizer.step()
                optimizer.zero_grad()
            # model.zero_grad()
            # obtain predictions and move them to cpu.
            preds = logits.argmax(dim=-1).cpu()
            stats[-1]["train_matches"] += (preds == batch[2].cpu()).sum().item()
            stats[-1]["train_tot"] += len(batch[0])
            stats[-1]["batch_losses"].append(loss.item())
            pbar.set_description(f"T: l: {np.mean(stats[-1]['batch_losses']):.3f} bl: {loss.item():.3f} sl: {scaled_loss.item():.3f} a: {(100*stats[-1]['train_matches']/stats[-1]['train_tot']):.2f}")
            # validation loss
            if ((step+1) % log_steps == 0) or ((step+1) == len(pbar)):
                val_acc, val_batch_losses = validate_cell_type_clf(
                    model, val_dataloader, **args
                )
                stats[-1]["val_batch_losses"] = val_batch_losses
                stats[-1]["val_loss"] = np.mean(val_batch_losses)
                stats[-1]["val_acc"] = val_acc # ultimate validation accuracy by the end of the epoch.
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_epoch_and_step = (epoch, step)
                    save_dict = {
                        "val_acc": val_acc,
                        "state_dict": model.state_dict(),
                        "best_val_acc": best_val_acc,
                        "best_epoch_and_step": best_epoch_and_step,
                    }
                    stats[-1]["best_epoch_and_step"] = best_epoch_and_step
                    stats[-1]["best_val_acc"] = best_val_acc # best validation accuracy for classification.
                    print(f"\n\nsaving \x1b[32;1mbest model\x1b[0m with val_acc: {100*best_val_acc:.3f}\n\n")
                    torch.save(save_dict, model_save_path)
                # update saved stats.
                stats_save_path = os.path.join(exp_folder, "stats.json")
                with open(stats_save_path, "w") as f:
                    json.dump(stats, f, indent=4)

# main method.
if __name__ == "__main__":
    args = get_args()
    # train_lm(**args)
    train_cell_type_clf(**args)