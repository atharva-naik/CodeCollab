#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code to visualize KG paths as decomposition tree structures.

import json
import pathlib
import treelib
from typing import *
from treelib import Node, Tree
from collections import defaultdict
# from treelib.node import DuplicatedNodeIdError

# use treelib for visualization
class TreeLibTreeBuilder:
    """Simple class to read JSON of KG paths and build
    a visualizable tree using `treelib`"""
    def __init__(self, path: str):
        self.path = path # path to KG paths JSON
        # the root name is the name of the library whose tutorials are indexed.
        self.root_name = " ".join(pathlib.Path(path).stem.split("_")).title()
        self.paths_dict = json.load(open(path))

    def build(self):
        # initialize new tree.
        self.built_tree = Tree()
        # create root node.
        self.built_tree.create_node(
            self.root_name, # display name
            self.root_name, # internal reference name.
        )
        for key in self.paths_dict:
            # key is basically a root to leaf path/path decompisition 
            # analogue for HTN (Hierarchical Task Networks).
            node_seq = [k.strip() for k in key.split("->")]
            for i, node in enumerate(node_seq):
                expanded_path = "->".join([self.root_name]+node_seq[:i+1])
                expanded_parent_path = "->".join([self.root_name]+node_seq[:i])
                try: 
                    self.built_tree.create_node(
                        node, expanded_path,
                        parent=expanded_parent_path,
                    )
                except Exception as e: pass

class TreeLibTreeBuilderFromData:
    """Simple class to visualize KG paths from Python dict data 
    (alread loaded instead of the class above that reads a JSOn)"""
    def __init__(self, data: dict):
        # the root name is the name of the library whose tutorials are indexed.
        self.root_name = "root"
        self.paths_dict = data

    def build(self):
        # initialize new tree.
        self.built_tree = Tree()
        # create root node.
        self.built_tree.create_node(
            self.root_name, # display name
            self.root_name, # internal reference name.
        )
        for key in self.paths_dict:
            # key is basically a root to leaf path/path decompisition 
            # analogue for HTN (Hierarchical Task Networks).
            node_seq = [k.strip() for k in key.split("->")]
            for i, node in enumerate(node_seq):
                expanded_path = "->".join([self.root_name]+node_seq[:i+1])
                expanded_parent_path = "->".join([self.root_name]+node_seq[:i])
                try: 
                    self.built_tree.create_node(
                        node, expanded_path,
                        parent=expanded_parent_path,
                    )
                except Exception as e: pass

# join KGs/KBs from various modules.
class ModuleJoiner:
    def __init__(self, list_of_modules: List[str]=["torch", "numpy", "sklearn", "pandas_toms_blog", "seaborn", "scipy"]):
        data = {}
        for module in list_of_modules:
            for key, value in json.load(open(f"./scrape_tutorials/KG_paths/{module}.json")).items():
                data[module+"->"+key] = value
        self.data = data

def prune_empty_nodes(KG: dict):
    pruned_KG = {}
    for path_key, cells in KG.items():
        pruned_KG[path_key] = []
        for cell in cells:
            if cell[0].strip() != "":
                pruned_KG[path_key].append(cell)
    
    return pruned_KG

# main
if __name__ == "__main__":
    data = ModuleJoiner().data # this is basically the unified, unpruned KG.
    with open("./scrape_tutorials/unified_KG_paths.json", "w") as f:
        json.dump(list(data.keys()), f, indent=4)
    with open("./scrape_tutorials/unified_KG.json", "w") as f:
        pruned_KG = prune_empty_nodes(data)
        json.dump(pruned_KG, f, indent=4)
    # print(data)
    visualizer = TreeLibTreeBuilderFromData(data)
    visualizer.build()
    # print(visualizer.build)
    # visualizer.built_tree.show()
    with open("./scrape_tutorials/unified_KG.txt", "w") as f:
        f.write(str(visualizer.built_tree))
