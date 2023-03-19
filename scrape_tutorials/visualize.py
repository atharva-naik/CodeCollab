#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code to visualize KG paths as decomposition tree structures.

import json
import pathlib
import treelib
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
                    # print(e)
                # # expanded name.
                # expanded_name = parent+"->"+node
                # # only add node if it hasn't been encountered/visited already (the node parent pair).
                # if expanded_name not in self.unique_node_parent_pair:
                #     self.unique_node_parent_pair[expanded_name] = True