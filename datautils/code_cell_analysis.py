import ast
import copy
import json
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm

builtins_set = set(['ArithmeticError',
 'AssertionError',
 'AttributeError',
 'BaseException',
 'BlockingIOError',
 'BrokenPipeError',
 'BufferError',
 'BytesWarning',
 'ChildProcessError',
 'ConnectionAbortedError',
 'ConnectionError',
 'ConnectionRefusedError',
 'ConnectionResetError',
 'DeprecationWarning',
 'EOFError',
 'Ellipsis',
 'EnvironmentError',
 'Exception',
 'False',
 'FileExistsError',
 'FileNotFoundError',
 'FloatingPointError',
 'FutureWarning',
 'GeneratorExit',
 'IOError',
 'ImportError',
 'ImportWarning',
 'IndentationError',
 'IndexError',
 'InterruptedError',
 'IsADirectoryError',
 'KeyError',
 'KeyboardInterrupt',
 'LookupError',
 'MemoryError',
 'ModuleNotFoundError',
 'NameError',
 'None',
 'NotADirectoryError',
 'NotImplemented',
 'NotImplementedError',
 'OSError',
 'OverflowError',
 'PendingDeprecationWarning',
 'PermissionError',
 'ProcessLookupError',
 'RecursionError',
 'ReferenceError',
 'ResourceWarning',
 'RuntimeError',
 'RuntimeWarning',
 'StopAsyncIteration',
 'StopIteration',
 'SyntaxError',
 'SyntaxWarning',
 'SystemError',
 'SystemExit',
 'TabError',
 'TimeoutError',
 'True',
 'TypeError',
 'UnboundLocalError',
 'UnicodeDecodeError',
 'UnicodeEncodeError',
 'UnicodeError',
 'UnicodeTranslateError',
 'UnicodeWarning',
 'UserWarning',
 'ValueError',
 'Warning',
 'ZeroDivisionError',
 '__IPYTHON__',
 '__build_class__',
 '__debug__',
 '__doc__',
 '__import__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 'abs',
 'all',
 'any',
 'ascii',
 'bin',
 'bool',
 'breakpoint',
 'bytearray',
 'bytes',
 'callable',
 'chr',
 'classmethod',
 'compile',
 'complex',
 'copyright',
 'credits',
 'delattr',
 'dict',
 'dir',
 'display',
 'divmod',
 'enumerate',
 'eval',
 'exec',
 'filter',
 'float',
 'format',
 'frozenset',
 'get_ipython',
 'getattr',
 'globals',
 'hasattr',
 'hash',
 'help',
 'hex',
 'id',
 'input',
 'int',
 'isinstance',
 'issubclass',
 'iter',
 'len',
 'license',
 'list',
 'locals',
 'map',
 'max',
 'memoryview',
 'min',
 'next',
 'object',
 'oct',
 'open',
 'ord',
 'pow',
 'print',
 'property',
 'range',
 'repr',
 'reversed',
 'round',
 'set',
 'setattr',
 'slice',
 'sorted',
 'staticmethod',
 'str',
 'sum',
 'super',
 'tuple',
 'type',
 'vars',
 'zip'])
def get_uniq_vars_and_funcs(cell_code_ast_root: ast.Module, 
                            imported_module_names: List[str]=[]) -> Dict[str, Dict[str, bool]]:
    """
    ### Input:
    - cell_code_ast_root: the root of the AST parse (ast.Module object)
    - (optional) imported_module_names: list of module names imported
    ### Method:
    Takes the root of the AST parse of the code and returns the list
    of unique variables and function calls in the order they appear in the AST
    ### Returns:
    - dict of variable names: the key is the name and the value tells if it is created/over-written in this cell (load vs store).
    - dict of function names: the key is the function name and the value tells if it is a module/object call.

    - caveat about variable names: can include modules as well as cell level 
    context might not be enough to distinguish between user defined identifiers 
    (variables/objects) and imported modules (they are filtered out if the user
    passes the list of import modules as the optional imported_module_names attribute).
    - caveat about variable appearance order: the order corresponds to AST traversal
    order and might differ from how a human would see the order of appearance (top to bottom, left to right)
    """
    uniq_vars_seq = {}
    uniq_func_seq = {}
    global builtins_set # skip builtin functions like 'print', 'abs' etc.
    imported_module_names = set(imported_module_names)
    skip_names = builtins_set.union(imported_module_names)
    for node in ast.walk(cell_code_ast_root):
        if isinstance(node, ast.Call):
            is_attr_call = True # is a module or object attribute
            if isinstance(node.func, ast.Name):
                func = node.func.id
                is_attr_call = False
            elif isinstance(node.func, ast.Attribute):
                func = node.func.attr
            if func in skip_names: continue 
            if func not in uniq_func_seq:
                uniq_func_seq[func] = is_attr_call
        # elif isinstance(node, ast.For): # variables generated in for loops.
        #     for target in node.
        # elif isinstance(node, ast.Assign): # always precedes the target(s) (which is (are) ast.Name types)
        #     for target in node.targets:
        #         if isinstance(target, ast.Name): # print(target.id)
        #             uniq_vars_seq[target.id] = True # is created/over-written in this cell
        elif isinstance(node, ast.Name):
            # print("how many times does 'pca' appear? (should be 3)", node.id)
            if node.id in skip_names: continue 
            if node.id not in uniq_vars_seq:
                uniq_vars_seq[node.id] = isinstance(node.ctx, ast.Store) # already needs to exist in a previous cell
                # NOTE: any variables created in this cell would be counted already as a target of an assignment expression
                # which is handled by the previous elif statement. So any new variables encountered in this statement
                # need to be previously defined in some other cell somewhere.
    # filter out any function names from the unique variable sequence (as they'll get double counted otherwise)
    pure_var_seq = {}
    for var, is_created_in_this_cell in uniq_vars_seq.items():
        if var not in uniq_func_seq:
            pure_var_seq[var] = is_created_in_this_cell
    uniq_vars_seq = pure_var_seq
    
    return {
        "vars": uniq_vars_seq, 
        "func": uniq_func_seq,
    }

def get_uniq_func_defs(code_cell_ast_root: ast.Module) -> List[str]:
    uniq_func_defs = {}
    for node in ast.walk(code_cell_ast_root):
        if isinstance(node, ast.FunctionDef):
            uniq_func_defs[node.name] = ''

    return list(uniq_func_defs.keys())

def get_line_count(code: str) -> int:
    num_lines = 0
    for line in code.split("\n"):
        line = line.strip()
        if line != "": num_lines += 1

    return num_lines

def strip_magic_from_code(cell_code: str): # and notebook commands (start with !).
    filt_lines = []
    for line in cell_code.split("\n"):
        lstrip = line.strip()
        if not(lstrip.startswith("%") or lstrip.startswith("!")):
            filt_lines.append(line)

    return "\n".join(filt_lines)

def strip_notebook_command(cell_code: str):
    filt_lines = []
    for line in cell_code.split("\n"):
        if not(line.strip().startswith("")):
            filt_lines.append(line)

    return "\n".join(filt_lines)

# program to test code construct sequence extraction from code cells. 
__TEST_PROGRAM = """
import json # import
from tqdm import tqdm # from

# func_def
def test(x):
# yield
    yield x+1

# class_def
class Test:
# pass
    pass

# func_def
def test2(x):
# return
    return x-1

# func_def
def test3(x):
# pass
    pass

async def test_async(x):
    print(x+1)
    
z = -1
path = 'test.txt'
while True: 
    if z > 0:
        break
    elif z < 0: 
        z += 1
    else: z = 0
    test(x)
    del x
    try: 
        x+1
        continue
    except: 
        x
        raise Exception(str(x))
    assert isinstance(x, bool)
    for i in range(z):
        print(z-1)
    async for i in range(z):
        print(z-1)
    with open(path, 'w') as f:
        json.dump(x, f, indent=4)
    async with open(path, 'w') as f:
        json.dump(x, f, indent=4)"""

def fix_python2_prints(code: str):
    fixed_code = []
    for line in code.split("\n"):
        try:
            first_token = line.strip().split()[0].strip()
            second_token = line.strip().split()[1].strip()
            if first_token == "print" and not(second_token.startswith("(")):
                line = line.replace("print ", "print(")+")"
        except: pass
        fixed_code.append(line)
    
    return "\n".join(fixed_code)

def ast_parse(code: str) -> ast.Module:
    code = strip_magic_from_code(code)
    try: root = ast.parse(code)
    except Exception as e:
        code = fix_python2_prints(code)
        root = ast.parse(code)

    return root

def get_code_construct_seq(code: str, api_sequence: List[str]=[]) -> List[str]:
    use_ref_api_seq, op_seq = False, []
    if len(api_sequence) != 0: use_ref_api_seq = True
    root = ast_parse(code)
    # ignore_node_types = (
    #     ast.Module, ast.Store, ast.withitem,
    #     ast.Attribute, ast.List, ast.Constant,
    #     ast.Name, ast.Load, ast.Expr, ast.Call,
    # )
    node_type_to_str = {
        ast.With: "with", ast.Assign: "assign",
        ast.For: "for", ast.FunctionDef: "func_def",
        ast.ClassDef: "class_def", ast.Delete: "delete",
        ast.Import: "import", ast.ImportFrom: "import_from",
        ast.Pass: "pass", ast.Return: "return", ast.Try: "try",
        ast.AsyncFor: "async_for", ast.AsyncWith: "async_with",
        ast.Assert: "assert", ast.AsyncFunctionDef: "async_func_def",
        ast.Break: "break", ast.Continue: "continue", ast.While: "while"
    }
    skipped = set()
    for node in ast.walk(root):
        # if isinstance(node, ast.Call):
        #     func = node.func
        #     if isinstance(func, ast.Name): func = func.id
        #     elif isinstance(func, ast.Attribute): func = func.attr
        #     if use_ref_api_seq:
        #         if func not in api_sequence: continue
        #         op_seq.append(func)
        #     else: op_seq.append(func)
        # elif 
        if isinstance(node, tuple(node_type_to_str.keys())):
            construct_type = node_type_to_str[type(node)]
            op_seq.append(construct_type)
        else: skipped.add(str(node))

    return op_seq
    # elif isinstance(node, ast.):        
def analyze_code_cell(cell_code: str, imported_module_names: List[str]=[],
                      func_set: set=set(), vars_set: set=set()) -> Tuple[Dict[str, int], Set[str], Set[str]]:
    """
    ### Input:
    - cell_code: the source code of the cell as a string.
    - (optional)
    - (optional) func_set: set of function names that are already defined by the previous cell
    - (optional) vars_set: set of variable names that are already defined by the previous cell
    
    ### Method:
    Takes the root of the AST parse of the code and returns the list
    of unique variables and function calls in the order they appear in the AST
    
    ### Returns:
    - stats: a dictionary with string keys and int values
    - func_set: update the func_set (optional input from above)
    - vars_set: update the vars_set (optional input from above)
    """
    f = open("ast_parse_error_codes.jsonl", "a")
    # cell_code = strip_magic_from_code(cell_code)
    try: root = ast_parse(cell_code)
    except Exception as e:
        print(e)
        print(cell_code)
        f.write(json.dumps({
            "code": cell_code,
            "error": str(e),
        })+"\n")
        # error in parsing AST
        stats = {
            "#vars need ctxt": 0,
            "has error": True,
            "#func need ctxt": 0,
            "#vars need ctxt": 0,
            "#func user def": 0,
            "#func call from prev ctxt": 0,
            "#lines": 0,
        }
        return stats, func_set, vars_set
    op = get_uniq_vars_and_funcs(root, imported_module_names)
    uniq_vars_seq = op["vars"]
    uniq_func_seq = op["func"]
    stats = {
        "#vars": len(uniq_vars_seq), 
        "#func": len(uniq_func_seq),
    }
    uniq_func_defs = get_uniq_func_defs(root)
    # number of variables that are not created/over-written in this cell 
    # (not on the left side of an assignment expression)
    stats["has error"] = False 
    stats["#vars need ctxt"] = len(uniq_vars_seq)-sum(list(uniq_vars_seq.values())) 
    # likely user defined functions (as they are not module or object member functions)
    stats["#func need ctxt"] = len(uniq_func_seq)-sum(list(uniq_func_seq.values()))
    stats["#vars has prev ctxt"] = 0
    stats["#func user def"] = len(uniq_func_defs)
    stats["#func call from prev ctxt"] = 0
    stats["#lines"] = get_line_count(cell_code)
    for var_name, is_created_in_this_cell in uniq_vars_seq.items():
        if is_created_in_this_cell: pass
            # vars_set.add(var_name)
        else: # not created in this cell.
            if var_name in vars_set:
                # print(var_name, vars_set) 
                stats["#vars has prev ctxt"] += 1
    for func_name, is_attr_call in uniq_func_seq.items():
        is_user_def = not(is_attr_call) # likely a user defined function.
        if is_user_def and func_name in func_set:
            stats["#func call from prev ctxt"] += 1
    # print("\x1b[32;1muniq_func_defs\x1b[0m", uniq_func_defs)
    # print("\x1b[32;1muniq_func_seq\x1b[0m", uniq_func_seq)
    # print("\x1b[32;1muniq_vars_seq\x1b[0m", uniq_vars_seq)
    for func_name, is_attr_call in uniq_func_seq.items(): 
        if not(is_attr_call): func_set.add(func_name)
    for func_name in uniq_func_defs: func_set.add(func_name)
    for var in uniq_vars_seq: vars_set.add(var)

    return stats, func_set, vars_set

def analyze_inst(inst: dict):
    context = inst["context"][::-1] # get the context in correct chronological order.
    imports = inst["imports"] # names of the imported modules.
    func_set = set()
    vars_set = set()
    stats = []
    summary = {} # summarize all the cell level stats for the notebook.
    cell_codes = [cell["code"] for cell in context if cell["cell_type"] == "code"]+[inst['code']]
    func_counts, vars_counts = [], []
    cells_using_ctxt = 0 # cells that actually use context from previous cells.
    has_missing_vars = False # some variables are never defined (very likely for most instances as they aren't complete notebooks)
    num_code = 1
    num_markdown = 0
    parsing_errors = 0
    for cell in context:
        cell_type = cell["cell_type"]
        if cell_type == "code":
            num_code += 1
        elif cell_type == "markdown":
            num_markdown += 1
    for cell_code in cell_codes:
        cell_stats, func_set, vars_set = analyze_code_cell(
            cell_code, imports,
            func_set, vars_set,
        )
        if cell_stats["has error"]: 
            parsing_errors += 1
            continue
        if cell_stats["#vars has prev ctxt"] or cell_stats["#func call from prev ctxt"]:
            cells_using_ctxt += 1
        if cell_stats["#vars need ctxt"] > cell_stats["#vars has prev ctxt"]:
            has_missing_vars = True
        func_counts.append(cell_stats["#func"])
        vars_counts.append(cell_stats["#vars"])
        cell_stats["code"] = cell_code
        cell_stats["func_set"] = copy.deepcopy(func_set)
        cell_stats["vars_set"] = copy.deepcopy(vars_set)
        stats.append(cell_stats)
    summary["#code"] = num_code
    try: summary["max func"] = np.max(func_counts)
    except Exception as e:
        summary["max func"] = 0
        print(e, func_counts)
    try: summary["max vars"] = np.max(vars_counts)
    except Exception as e:
        summary["max vars"] = 0
        print(e, func_counts)
    summary["mean func"] = round(np.mean(func_counts), 2)
    summary["mean vars"] = round(np.mean(vars_counts), 2)
    summary["#markdown"] = num_markdown
    summary["cells using ctxt"] = cells_using_ctxt
    summary["has missing vars"] = has_missing_vars
    summary["#parsing errors"] = parsing_errors

    return stats, summary

def analyze_all_data(data: Dict[str, dict]) -> List[Dict[str, Union[float, int, bool]]]:
    analysis_summary = []
    for id, inst in tqdm(data.items()):
        _, summary = analyze_inst(inst)
        summary["id"] = id
        analysis_summary.append(summary)

    return analysis_summary

def generate_final_stats(summ: List[Dict[str, Union[float, int, bool]]]) -> Dict[str, Union[float, int, bool]]:
    summ_df = pd.DataFrame(summ)
    return {
        "avg. #code": round(summ_df["#code"].mean(), 2),
        "avg. #markdown": round(summ_df["#markdown"].mean(), 2),
        "max func": summ_df["max func"].max(),
        "max vars": summ_df["max vars"].max(),
        "mean func": summ_df["mean func"].mean(),
        "mean vars": summ_df["mean vars"].mean(),
        "cell using ctxt": summ_df["cells using ctxt"].sum(),
        "percent cells using ctxt": round(100*(summ_df["cells using ctxt"].sum()/summ_df["#code"].sum()), 2),
        "#parsing errors": summ_df["#parsing errors"].sum(),
        "percent parsing errors": round(100*(summ_df["#parsing errors"].sum()/summ_df["#code"].sum()), 2),
        "incomplete nbs": summ_df["has missing vars"].sum()/len(summ_df), 
        "percent incomplete nbs": round(100*(summ_df["has missing vars"].sum()/len(summ_df)), 2)
    }

if __name__ == "__main__":
    path = "./data/juice-dataset/sampled_juice_train.json"
    data = json.load(open(path))
    open("ast_parse_error_codes.jsonl", "w") # clear errors log.
    analysis_summary = analyze_all_data(data)
    # analysis_df = pd.DataFrame(analysis_summary)
    # analysis_df.to_excel("./analysis/sampled_juice_analysis_summary.xlsx")