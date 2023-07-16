
# script for chunking FCDS code cells (that implement the same intent)
import ast
import sys
import json
import builtins
from typing import *
from fuzzywuzzy import fuzz
from collections import defaultdict
from datautils.code_cell_analysis import remove_comments_and_docstrings

BUILTIN_TYPES =  [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]
BUILTIN_TYPE_INIT_CALLS = [f"{i.__name__}()" for i in BUILTIN_TYPES]
def get_variables_from_targets(targets: List[Union[ast.Tuple, ast.Name, ast.List]]) -> Dict[str, None]:
    variable_names = {}
    for tuple_or_name in targets:
        if isinstance(tuple_or_name, ast.Name):
            variable_names[tuple_or_name.id] = None
        elif isinstance(tuple_or_name, (ast.Tuple, ast.List)):
            variable_names.update(
                get_variables_from_targets(tuple_or_name.elts)
            )

    return variable_names

def chunk_code_to_variable_blocks(code: str) -> Dict[str, str]:
    code = code.strip()
    root = ast.parse(code)
    VARIABLES = {} # collect important variables (that occur on the left side of an assignment statement)
    for node in ast.walk(root):
        if isinstance(node, (ast.Assign)):
            # print(ast.unparse(node))
            # print(ast.unparse(node.targets), ":", node.targets)
            variable_names = get_variables_from_targets(node.targets)
            VARIABLES.update(variable_names)
    variable_block_limits = {v: [sys.maxsize, 0] for v in VARIABLES}
    # associate with each variable the chunks of code relevant to it.

    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            if node.id not in variable_block_limits: continue
            top, bottom = variable_block_limits[node.id]
            top = min(top, node.lineno)
            bottom = max(bottom, node.lineno)
            variable_block_limits[node.id][0] = top
            variable_block_limits[node.id][1] = bottom
    # remove variables with empty blocks:
    variable_full_blocks = {}
    for v, (top, bottom) in variable_block_limits.items():
        if top == bottom: continue
        variable_full_blocks[v] = "\n".join(code.split("\n")[top-1:bottom])

    return variable_full_blocks

sample_code = '''def movie_count_by_genre(movies, genres):
    """
    Count the number of movies in each movie genre.

    args:
        movies (pd.DataFrame) : Dataframe containing movie attributes
        genres (List[str]) :  the list of movie genres

    return:
        Dict[str, Dict[int, int]]  : a nested mapping from movie genre to year to number of movies in that year
    """
    result = dict()
    def convert(x):
        return int(x[-4:])
    movies = movies.dropna(subset=['release_date'])
    movies["release_date"] = movies["release_date"].apply(convert)
    for genre in genres:
        genre_dict = dict()
        genre_df = movies[movies[genre] == 1]
        count = genre_df.groupby(['release_date'], group_keys=True)['release_date'].count().reset_index(name="count")
        for i in range(len(count)):
            genre_dict[int(count.iloc[[i]]["release_date"])] = int(count.iloc[[i]]["count"])
        result[genre] = genre_dict

    return result'''

CODE_CONSTRUCTS = (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, 
                   ast.With, ast.If, ast.IfExp, ast.Call, ast.Assign)

def find_overlapping_block(code: str, lineno: int, end_lineno: int, 
                           block_limits: Dict[str, str]) -> List[str]:
    chunk_hierarchy = []
    for limit_str, block_code in block_limits.items():
        block_lineno = int(limit_str.split("::")[0])
        block_end_lineno = int(limit_str.split("::")[1])
        if block_lineno <= lineno and end_lineno <= block_end_lineno and fuzz.token_set_ratio(code.strip(), block_code.strip()) >= 0.95: 
            chunk_hierarchy.append(block_code)

    return chunk_hierarchy

def sort_function(l, c):
    return 10000*l+c

def sort_chunk_tree(chunk_tree: dict, node_to_loc_mapping: Dict[str, Tuple[int, int, int, int]]):
    """recursively sort a chunk tree based on code line numbers and column offsets."""
    sorted_chunk_tree = {}
    for chunk, subtree in chunk_tree.items():
        sorted_chunk_tree[chunk] = sort_chunk_tree(subtree, node_to_loc_mapping)
    sorted_chunk_tree = {
        chunk: subtree for chunk, subtree in sorted(
            sorted_chunk_tree.items(), reverse=False, 
            key=lambda x: sort_function(*node_to_loc_mapping[x[0]])
        )
    }

    return sorted_chunk_tree

def extract_op_chunks(code: str):
    global BUILTIN_TYPE_INIT_CALLS
    # remove comments and docstrings
    code = remove_comments_and_docstrings(code)
    global CODE_CONSTRUCTS
    root = ast.parse(code)
    nodes = {}
    chunk_tree = {}
    block_limits = {}
    flat_chunk_list = {}
    node_to_loc_map = {}
    for node in ast.walk(root):
        if isinstance(node, CODE_CONSTRUCTS):
            sub_code = ast.unparse(node)
            if sub_code in BUILTIN_TYPE_INIT_CALLS: continue
            block_limits[f"{node.lineno}::{node.end_lineno}"] = sub_code
            nodes[sub_code] = node
            node_to_loc_map[sub_code] = (node.lineno, node.col_offset)
    # sort by block lengths in descending order.
    block_limits = {k: v for k,v in sorted(block_limits.items(), key=lambda x: len(x[1]), reverse=True)}
    nodes = {k: v for k,v in sorted(nodes.items(), key=lambda x: x[0], reverse=True)}
    for node_code, node in nodes.items():
        chunk_hierarchy = find_overlapping_block(
            node_code, node.lineno, 
            node.end_lineno, block_limits,
        )
        root = chunk_tree
        # print(chunk_hierarchy)
        for sub_code in chunk_hierarchy:
            if sub_code not in flat_chunk_list:
                flat_chunk_list[sub_code] = ""
            if sub_code not in root:
                root[sub_code] = defaultdict(lambda: {})
            root = root[sub_code]
    chunk_tree = sort_chunk_tree(chunk_tree, node_to_loc_map)
    flat_chunk_list = list(flat_chunk_list.keys())
                
    return chunk_tree, flat_chunk_list

def print_chunks(chunks: dict, level=1):
    for chunk, sub_chunks in chunks.items():
        print("#"*level+"\n"+chunk)
        print()
        print_chunks(sub_chunks, level=level+1)
# main
# if __name__ == "__main__":
#     eg_code = """def predict_user_user(X, W, user_means, eps=1e-12):
#     X_normalized = X-user_means.reshape(-1,1)
#     X_mask = np.where(X>0, 1,0)
#     num = W.dot(X_normalized*X_mask)
#     W_non_zero = W.dot(X_mask)
#     term = num / (eps + W_non_zero)

#     return user_means.reshape(-1,1) + term"""
#     variable_blocks = chunk_code_to_variable_blocks(eg_code)
#     for v, block in variable_blocks.items():
#         print("\n------------------------------------")
#         print(f"\x1b[34;1mVARIBLE BLOCK for: {v}\x1b[0m")
#         print(block)
#         print("------------------------------------\n")
if __name__ == "__main__":
    chunk_tree, flat_chunk_list = extract_op_chunks(sample_code)
    # print_chunks(chunks)
    print(json.dumps(chunk_tree, indent=4))
    print(json.dumps(flat_chunk_list, indent=4))
    