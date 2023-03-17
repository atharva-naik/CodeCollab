#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

proc_preds = {}
with open("./modelout/predictions/predict_notebook.txt", "r") as f:
    for line in f:
        rec = line.strip().split("\t")
        rec = [i.replace("<NULL>", " ").replace("<s>", " ").replace("</s>", " ").strip() for i in rec]
        # print(len(rec))
        proc_preds[rec[0]] = rec[1]
with open("./modelout/predictions/predict_notebook_proc.json", "w") as f:
    json.dump(proc_preds, f, indent=4)