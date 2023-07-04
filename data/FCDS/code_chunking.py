
# script for chunking FCDS code cells (that implement the same intent)
import ast
import sys
from typing import *
# from datautils.code_cell_analysis import remove

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

# main
if __name__ == "__main__":
    eg_code = """def predict_user_user(X, W, user_means, eps=1e-12):
        X_normalized = X-user_means.reshape(-1,1)
        X_mask = np.where(X>0, 1,0)
        num = W.dot(X_normalized*X_mask)
        W_non_zero = W.dot(X_mask)
        term = num / (eps + W_non_zero)

        return user_means.reshape(-1,1) + term"""
    variable_blocks = chunk_code_to_variable_blocks(eg_code)
    for v, block in variable_blocks.items():
        print("\n------------------------------------")
        print(f"\x1b[34;1mVARIBLE BLOCK for: {v}\x1b[0m")
        print(block)
        print("------------------------------------\n")