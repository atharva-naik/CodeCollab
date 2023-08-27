
# script for chunking FCDS code cells (that implement the same intent)
import ast
import sys
import copy
import json
import builtins
from typing import *
from fuzzywuzzy import fuzz
from collections import defaultdict
from datautils.code_cell_analysis import remove_comments_and_docstrings

BUILTIN_TYPES =  [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]
BUILTIN_TYPE_NAMES = [i.__name__ for i in BUILTIN_TYPES]
BUILTIN_TYPE_INIT_CALLS = [f"{i.__name__}()" for i in BUILTIN_TYPES]
AST_CONSTRUCTS = [ele.upper() for ele in dir(ast) if type(getattr(ast, ele)) == type]

def process_tuple_elt(tuple_or_name, variable_names: Dict[str, None]):
    assert isinstance(tuple_or_name, (ast.Tuple, ast.Name, ast.Subscript, ast.List, ast.Attribute, ast.Call, ast.Constant)), f"{ast.unparse(tuple_or_name)} is a {type(tuple_or_name).__name__}"
    if isinstance(tuple_or_name, ast.Name):
        variable_names[tuple_or_name.id] = None
    elif isinstance(tuple_or_name, (ast.Tuple, ast.List)):
        variable_names.update(
            get_variables_from_targets(tuple_or_name.elts)
        )
    elif isinstance(tuple_or_name, ast.Subscript):
        process_tuple_elt(tuple_or_name.value, variable_names)
    elif isinstance(tuple_or_name, ast.Attribute):
        # print(ast.unparse(tuple_or_name), 
        #       tuple_or_name.attr, 
        #       ast.unparse(tuple_or_name.value))
        process_tuple_elt(tuple_or_name.value, variable_names)

    return variable_names

def get_variables_from_targets(targets: List[Union[ast.Tuple, ast.Name, ast.Subscript, ast.List, ast.Attribute, ast.Call, ast.Constant]], return_list: bool=False) -> Dict[str, None]:
    variable_names = {}
    if isinstance(targets, Iterable):
        for tuple_or_name in targets:
            process_tuple_elt(tuple_or_name, variable_names)
    else: process_tuple_elt(targets, variable_names)
    if return_list: return list(variable_names.keys())

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
EXPANDED_CODE_CONSTRUCTS = (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, 
                            ast.With, ast.If, ast.IfExp, ast.Call, ast.Assign,
                            ast.ListComp, ast.DictComp, ast.SetComp, ast.Return,
                            ast.Subscript)

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

class ChunkRepr:
    def __init__(self, chunk_fields_and_values: dict):
        self.fields_and_values = chunk_fields_and_values

    def todict(self):
        return self.fields_and_values

    def __lt__(self, other):
        if self.fields_and_values["META_lineno"] < other.fields_and_values["META_lineno"]:
           return True
        elif self.fields_and_values["META_lineno"] == other.fields_and_values["META_lineno"]:
            if self.fields_and_values["META_end_lineno"] > other.fields_and_values["META_end_lineno"]:
                return True
            elif self.fields_and_values["META_end_lineno"] == other.fields_and_values["META_end_lineno"]:
                if self.fields_and_values["META_col_offset"] < other.fields_and_values["META_col_offset"]:
                    return True
                elif self.fields_and_values["META_col_offset"] == other.fields_and_values["META_col_offset"]:
                    return self.fields_and_values["META_end_col_offset"] > other.fields_and_values["META_end_col_offset"]
                else: return False
            else: return False
        else: return False

    def __eq__(self, other):
        return self.fields_and_values["META_lineno"] == other.fields_and_values["META_lineno"] and self.fields_and_values["META_end_lineno"] == other.fields_and_values["META_end_lineno"] and self.fields_and_values["META_col_offset"] == other.fields_and_values["META_col_offset"] and self.fields_and_values["META_end_col_offset"] == other.fields_and_values["META_end_col_offset"]

    def __getitem__(self, field):
        return self.fields_and_values[field]

def extract_plan_op_chunks_v2(code):
    nodecode2id = {}
    codecons2id = {}
    root = ast.parse(code)
    id = 0
    consid = 0
    for node in ast.walk(root):
        nodecode = ast.unparse(node)
        if isinstance(node, EXPANDED_CODE_CONSTRUCTS) and nodecode not in BUILTIN_TYPE_INIT_CALLS:
            codecons2id[nodecode] = consid
            consid += 1
        nodecode2id[nodecode] = id
        id += 1 
    root = ast.parse(code)
    chunks = []
    for node in ast.walk(root):
        nodecode = ast.unparse(node)
        if not isinstance(node, EXPANDED_CODE_CONSTRUCTS) or nodecode in BUILTIN_TYPE_INIT_CALLS: continue
        chunk = {
            "META_chunktype": type(node).__name__,
            "META_lineno": node.lineno,
            "META_end_lineno": node.end_lineno, 
            "META_col_offset": node.col_offset, 
            "META_end_col_offset": node.end_col_offset,
            "META_code": nodecode,
            "META_id": codecons2id[nodecode]
        }
        for field, value in node.iter_fields():
            if field == "type_comment": field = "META_type_comment"
            if isinstance(value, list):
                proc_value = []
                for member in value:
                    if isinstance(member, EXPANDED_CODE_CONSTRUCTS) and ast.unparse(value) not in BUILTIN_TYPE_INIT_CALLS:
                        proc_value.append(codecons2id[ast.unparse(member)])
                    elif isinstance(member, ast.AST): continue
                    else: proc_value.append(member)
                value = proc_value
            elif isinstance(value, EXPANDED_CODE_CONSTRUCTS) and ast.unparse(value) not in BUILTIN_TYPE_INIT_CALLS: 
                value = codecons2id[ast.unparse(value)]
            elif isinstance(value, ast.AST): continue
            else: value = str(value)
            assert isinstance(value, (int, str, list)), f"{value}, type: {type(value)}"
            chunk[field] = value
        chunks.append(chunk)

    return nodecode2id, codecons2id, chunks

def validate_plan_op_annot_chunks(chunks):
    for chunk in chunks:
        assert "META_plan_op" in chunk, f"len(chunks)={len(chunks)} {chunk['META_code']} {chunk.keys()}"
        assert "META_plan_op_score" in chunk, f"len(chunks)={len(chunks)} {chunk['META_code']} {chunk.keys()}"

def validate_plan_op_annot_submissions(task_subs):
    for i, sub in enumerate(task_subs):
        try: validate_plan_op_annot_chunks(sub["chunks"])
        except AssertionError as e:
            print("Index[", i, "]")
            print(e)
            return

def sort_and_remap_chunks(chunks: List[dict]):
    chunks = [chunk.todict() for chunk in sorted([ChunkRepr(chunk) for chunk in chunks])]
    new_id_mapping = {}
    for i, chunk in enumerate(chunks):
        if chunk["META_id"] not in new_id_mapping:
            new_id_mapping[chunk["META_id"]] = i
    for chunk in chunks:
        chunk["META_id"] = new_id_mapping[chunk["META_id"]]
        for key, value in chunk.items():
            if key.startswith("META"): continue
            if isinstance(value, list):
                chunk[key] = [new_id_mapping[v] for v in value]
            elif isinstance(value, int):
                chunk[key] = new_id_mapping[value]

    return chunks

def get_kwarg_from_args(args: ast.arguments) -> List[ast.arg]:
    if args.kwarg: return [args.kwarg]
    return []

def get_vararg_from_args(args: ast.arguments) -> List[ast.arg]:
    if args.vararg: return [args.vararg]
    return []

def list_funcdef_arguments(funcdef: ast.FunctionDef):    
    kwarg = get_kwarg_from_args(funcdef.args)
    vararg = get_vararg_from_args(funcdef.args)

    return [a.arg for a in funcdef.args.args+funcdef.args.posonlyargs+funcdef.args.kwonlyargs+kwarg+vararg]

def extract_arguments_from_code(code: str) -> List[str]:
    """extract arguments from string from version implementing a function"""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef):
            return list_funcdef_arguments(node)
    return []

def modify_complex_return_statements_in_code(code: str) -> str:
    """edit very complex return statements occuring in code.
    i.e. statements which aren't ast.Tuple or ast.Name.
    
    only cares about the top level function's return statement."""
    root = ast.parse(code)
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            for index, stmt in enumerate(node.body):
                if not isinstance(stmt, ast.Return): continue
                if not isinstance(stmt.value, ast.Name):
                    node.body.insert(index, ast.parse(f"__SIMPLIFIED_RETURN_VARIABLE = {ast.unparse(stmt.value)}"))
                    stmt.value = ast.Name(id="__SIMPLIFIED_RETURN_VARIABLE")
    code = ast.unparse(root)
    
    return code

def extract_returned_variables_from_code(code: str) -> List[str]:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    # print(ast.unparse(stmt), stmt.value)
                    return_code = ast.unparse(stmt)
                    return [node.id for node in ast.walk(ast.parse(return_code)) if isinstance(node, ast.Name)]
    return [] 

def process_field(field_value: ast.AST) -> str:
    if isinstance(field_value, list):
        all_vars = []
        for val in field_value:
            output = process_field(val)
            if isinstance(output, str): output = [output]
            for var in output:
                if var in all_vars: continue
                all_vars.append(var)

        return "["+', '.join(all_vars) +"]"
    elif isinstance(field_value, ast.Subscript):
        return f"SUBSCRIPT(slice={process_field(field_value.slice)}, value={process_field(field_value.value)})"
    elif isinstance(field_value, ast.Name): return field_value.id
    elif isinstance(field_value, (ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Invert, ast.UAdd, ast.USub, ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv, ast.Load, ast.Store, ast.Del)):
        return type(field_value).__name__.upper()
    elif isinstance(field_value, str): return field_value
    elif field_value is None: return "NONE"
    elif isinstance(field_value, ast.arguments):
        posonlyargs = field_value.posonlyargs if field_value.posonlyargs is not None else []
        kwonlyargs = field_value.kwonlyargs if field_value.kwonlyargs is not None else []
        vararg = field_value.vararg.arg if field_value.vararg is not None else None
        kwarg = field_value.kwarg.arg if field_value.kwarg is not None else None
        return f"ARGUMENTS(args=[{', '.join([arg.arg for arg in field_value.args])}], kwarg={kwarg}, kwonlyargs=[{', '.join([arg.arg for arg in kwonlyargs])}], posonlyargs=[{', '.join([arg.arg for arg in posonlyargs])}], vararg={vararg})"
    elif isinstance(field_value, ast.Compare):
        # print(field_value.left)
        return f"COMPARE(left={process_field(field_value.left)}, ops=[{', '.join([type(op).__name__.upper() for op in field_value.ops])}], comparators={process_field(field_value.comparators)})"
    elif isinstance(field_value, ast.BinOp):
        return f"BINOP(op={type(field_value.op).__name__.upper()}, left={process_field(field_value.left)}, right={process_field(field_value.right)})"
    elif isinstance(field_value, ast.UnaryOp):
        return f"UNARYOP(op={type(field_value.op).__name__.upper()}, operand={process_field(field_value.operand)})"
    elif isinstance(field_value, ast.BoolOp): # print(field_value.values)
        # print(ast.unparse(field_value), field_value.op, field_value.values)
        return f"BOOLOP(op={process_field(field_value.op)}, values={process_field(field_value.values)})"
    elif isinstance(field_value, ast.withitem):
        assert not(isinstance(field_value.optional_vars, Iterable)), f"withitem.optional_vars is an iterable: {field_value}"
        return f"WITHITEM(context_expr={process_field(field_value.context_expr)}, optional_vars={field_value.optional_vars.id}])"
    else: return get_variables_from_targets(field_value, return_list=True)

class VarNameObfuscater(ast.NodeTransformer):
    def __init__(self, input_list: List[str]=[],
                 output_list: List[str]=[],
                 exclude_list: List[str]=[]):
        super().__init__()
        self.exclude_list = exclude_list+BUILTIN_TYPE_NAMES
        self.input_list = input_list
        self.output_list = output_list
        self.call_names = {}
        self.calls_collected = False
        # print(self.exclude_list)
        self.obf_mapping = {}
        self.inp_mapping = {input_list[i]: f"INP{i}" for i in range(len(input_list))}
        self.out_mapping = {output_list[i]: f"OUT{i}" for i in range(len(output_list))}
        self.out_mapping.update(self.inp_mapping)
        self.declared_vars = {}

    def custom_visit(self, root: ast.AST):
        self.calls_collected = False
        self.call_names = {}
        root = self.visit(root)
        self.calls_collected = True
        self.declared_vars = {}

        return self.visit(root)

    def visit_Call(self, node: ast.Call) -> Any:
        if not self.calls_collected:
            if isinstance(node.func, ast.Name):
                self.call_names[node.func.id] = True

        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if self.calls_collected and node.id not in self.call_names:
            if node.id in self.input_list:
                inp_var = self.inp_mapping.get(node.id)
                if inp_var is None:
                    inp_var = f"INP{len(self.inp_mapping)}"
                    self.inp_mapping[node.id] = inp_var
                node.id = self.inp_mapping[node.id]   
            elif node.id in self.output_list:
                node.id = self.out_mapping[node.id]             
            elif node.id not in self.exclude_list:
                if isinstance(node.ctx, ast.Store):
                    self.declared_vars[node.id] = True
                elif isinstance(node.ctx, ast.Load) and not(self.declared_vars.get(node.id, False)):
                    node.id = "__MODULE_"+node.id
                    return super().generic_visit(node) 
                obf_var = self.obf_mapping.get(node.id)
                if obf_var is None:
                    obf_var = f"VAR{len(self.obf_mapping)}"
                    self.obf_mapping[node.id] = obf_var
                node.id = obf_var 

        return super().generic_visit(node)

def extract_ops_from_statement(stmt: ast.AST) -> List[str]:
    """extract ops (function calls, list/dict/set comprehensions etc.)"""
    ops = []
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            call_name = ast.unparse(node.func).split(".")[-1].strip()
            if call_name in BUILTIN_TYPE_NAMES: continue
            ops.append(call_name)
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.DictComp)):
            ops.append(type(node).__name__.upper())

    return ops[::-1]
    # [ast.unparse(node.func).split(".")[-1].strip() for node in ast.walk(stmt) if isinstance(node, (ast.Call, ast.ListComp))][::-1]    

# single function subprogram decomposer:
class FunctionSubProgramDecomposer:
    """Class to do Sub program decompositions of
    function defintions."""
    def __init__(self):
        self.init_io = (None, None)
        self.vocab = AST_CONSTRUCTS + ["END"+const for const in AST_CONSTRUCTS] + [f"VAR{i}" for i in range(100)] + [f"INP{i}" for i in range(100)] + ["ELSE", "FINALLY", "EXCEPT"] + [f"OUT{i}" for i in range(100)] + ["[STEP]", "[I]", "[O]", "[ops]"]#+["[T]"]

    def update_first_goal(self, first_goal, subgoal_seq) -> bool:
        if not(first_goal): subgoal_seq.append("NEWLINE")
        else: first_goal = False

        return first_goal

    def extract_subgoal_inputs(self, STMT) -> List[str]:
        subgoal_inputs = []
        for node in ast.walk(STMT):
            if not isinstance(node, ast.Name): continue 
            if not(node.id.startswith("VAR") or node.id.startswith("OUT") or node.id.startswith("INP") or node.id.startswith("__MODULE_")): continue
            if node.id in subgoal_inputs: continue
            if node.id.startswith("__MODULE_"): 
                subgoal_inputs.append(node.id[len("__MODULE_"):])
            else: subgoal_inputs.append(node.id)

        return subgoal_inputs

    def strip_module_tag(self, var_seq: List[str]):
        for i in range(len(var_seq)):
            if var_seq[i].startswith("__MODULE_"):
                var_seq[i] = var_seq[i][len("__MODULE_"):]
        return var_seq

    def extract_subgoals(self, body: List[ast.AST]):
        first_goal = True
        subgoal_seq = []
        for stmt in body:
            if stmt is None: continue
            elif isinstance(stmt, (ast.Assign, ast.Expr, ast.AugAssign)):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call): 
                    # TODO: # likely an inplace call (test this assumption.)
                    subgoal_outputs = get_variables_from_targets(stmt.value.func, return_list=True)
                elif isinstance(stmt, ast.AugAssign): subgoal_outputs = get_variables_from_targets(stmt.target, return_list=True)
                elif isinstance(stmt, ast.Expr): subgoal_outputs = []
                else: subgoal_outputs = get_variables_from_targets(stmt.targets, return_list=True)
                ops = extract_ops_from_statement(stmt)
                if isinstance(stmt, ast.AugAssign):
                    ops.append(type(stmt.op).__name__.upper())
                subgoal_inputs = self.extract_subgoal_inputs(stmt.value)
                subgoal_seq.append({
                    "I": subgoal_inputs, "O": self.strip_module_tag(subgoal_outputs), 
                    "ops": ops, "code": ast.unparse(stmt)
                })

            elif isinstance(stmt, ast.Name):
                # first_goal = self.update_first_goal(first_goal, subgoal_seq)
                subgoal_seq.append({
                    "I": [stmt.id], "O": self.strip_module_tag([stmt.id]), 
                    "ops": [], "code": ast.unparse(stmt)
                })

            elif isinstance(stmt, ast.BoolOp):
                subgoal_seq.append(f"BOOLOP(op={process_field(stmt.op)}, values={process_field(stmt.values)})")

            elif isinstance(stmt, ast.Subscript):
                subgoal_seq.append(f"SUBSCRIPT(slice={process_field(stmt.slice)}, value={process_field(stmt.value)})")

            elif isinstance(stmt, ast.Global):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                subgoal_seq.append(f"GLOBAL(names=[{', '.join(stmt.names)}])")

            elif isinstance(stmt, ast.Call):
                # first_goal = self.update_first_goal(first_goal, subgoal_seq)
                subgoal_outputs = get_variables_from_targets(stmt.func, return_list=True)
                ops = extract_ops_from_statement(stmt)
                
                subgoal_inputs = []
                for node in stmt.args:
                    for input_var in self.extract_subgoal_inputs(node):
                        if input_var in subgoal_inputs: continue
                        subgoal_inputs.append(input_var)
                
                subgoal_seq.append({
                    "I": subgoal_inputs, "O": self.strip_module_tag(subgoal_outputs), 
                    "ops": ops, "code": ast.unparse(stmt)
                })

            elif (hasattr(stmt, "_fields") and "body" in stmt._fields) or isinstance(stmt, (ast.Raise, ast.Delete)):
                # first_goal = self.update_first_goal(first_goal, subgoal_seq)
                construct = type(stmt).__name__.upper()
                args_string = []
                for k in stmt._fields:
                    if k in ["body", "type_comment", "orelse", "exc", "targets", "handlers", "finalbody"]: continue
                    try: args_string.append(f"{k}={process_field(getattr(stmt, k))}")
                    except AssertionError as e:
                        print(f"\x1b[31;1mAssertionError:\x1b[0m {e}")
                        print(construct)
                        print(k)
                        print(getattr(stmt, k))
                        exit("Assert triggered within process_field")
                # add the construct
                args_string = ", ".join(args_string)
                subgoal_seq.append(construct+"("+args_string+")")

                # add the body and other body like fields.
                for field in ["body", "orelse", "exc", "targets", "handlers", "finalbody"]:
                    if field not in stmt._fields: continue
                    field_value = getattr(stmt, field)
                    if not isinstance(field_value, list) : field_value = [field_value]
                    if len(field_value) == 0: continue
                    if field == "orelse": subgoal_seq.append("ELSE")
                    elif field_value == "handlers": subgoal_seq.append("EXCEPT")
                    elif field_value == "finalbody": subgoal_seq.append("FINALLY")
                    subgoal_seq += self.extract_subgoals(field_value)

                # end the construct and add it to the vocab if needed.
                subgoal_seq.append("END"+construct)
                if "END"+construct not in self.vocab:
                    self.vocab.append("END"+construct)
            elif isinstance(stmt, ast.Import):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                # print(f"IMPORT(alias=[{', '.join([f'ALIAS(name={alias.name}, asname={alias.asname})' for alias in stmt.names])}])")
                return f"IMPORT(alias=[{', '.join([f'ALIAS(name={alias.name}, asname={alias.asname})' for alias in stmt.names])}])"
            elif isinstance(stmt, ast.ImportFrom):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                # print(f"IMPORTFROM(module={stmt.module}, alias=[{', '.join([f'ALIAS(name={alias.name}, asname={alias.asname})' for alias in stmt.names])}], level={stmt.level})")
                return f"IMPORTFROM(module={stmt.module}, alias=[{', '.join([f'ALIAS(name={alias.name}, asname={alias.asname})' for alias in stmt.names])}], level={stmt.level})"
            elif isinstance(stmt, ast.Assert):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                subgoal_seq.append(f"ASSERT(test={process_field(stmt.test)}, msg={process_field(stmt.msg)})")
            elif isinstance(stmt, (ast.Return, ast.Continue, ast.Pass, ast.Break, ast.Del, ast.Load, ast.Store)):
                first_goal = self.update_first_goal(first_goal, subgoal_seq)
                subgoal_seq.append(type(stmt).__name__.upper())
            else: 
                print(f"Unhandled consruct of type: {type(stmt).__name__.upper()}")
                print(stmt, ast.unparse(stmt))
                for field in stmt._fields:
                    print(field+":", getattr(stmt, field))
                raise NotImplementedError

        return subgoal_seq

    def extract_subgoals_from_subprograms(self, body: List[ast.AST], inputs: List[str], 
                                          outputs: List[str], target: List[str]):
        first_goal = True
        subgoal_seq = []
        subgoal_inputs = copy.deepcopy(inputs)
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                if not(first_goal): subgoal_seq.append("NEWLINE")
                else: first_goal = False
                subgoal_outputs = get_variables_from_targets(stmt.targets, return_list=True)
                ops = extract_ops_from_statement(stmt)
                _subgoal_inputs = []
                for node in ast.walk(stmt.value):
                    if not isinstance(node, ast.Name): continue 
                    if not(node.id.startswith("VAR") or node.id.startswith("OUT") or node.id.startswith("INP")): continue
                    if node.id in _subgoal_inputs: continue
                    _subgoal_inputs.append(node.id)
                subgoal_seq.append({
                    "I": _subgoal_inputs, "O": self.strip_module_tag(subgoal_outputs), 
                    # "I": copy.deepcopy(subgoal_inputs), "O": subgoal_outputs, 
                    "T": target, "ops": ops, "code": ast.unparse(stmt)
                })
                # for op in subgoal_outputs:
                #     if op not in subgoal_inputs:
                #         subgoal_inputs.append(op)
            elif isinstance(stmt, (ast.Call, ast.Expr)):
                if isinstance(stmt, ast.Expr): 
                    # basically dealing with inplace function calls.
                    if isinstance(stmt.value, ast.Call):
                        stmt = stmt.value # skip expressions that are not function calls.
                    else: continue
                if not(first_goal): subgoal_seq.append("NEWLINE")
                else: first_goal = False
                ops = [ast.unparse(node.func).split(".")[-1].strip() for node in ast.walk(stmt) if isinstance(node, ast.Call)][::-1]
                subgoal_outputs = get_variables_from_targets(stmt.func, return_list=True)
                subgoal_seq.append({
                    "I": copy.deepcopy(subgoal_inputs), "O": self.strip_module_tag(subgoal_outputs), 
                    "T": target, "ops": ops, "code": ast.unparse(stmt)
                })
            elif "body" in stmt._fields:
                construct = type(stmt).__name__.upper()
                args_string = []
                for k in stmt._fields:
                    if k in ["body", "type_comment", "orelse"]: continue
                    try: args_string.append(f"{k}={process_field(getattr(stmt, k))}")
                    except AssertionError:
                        print(construct)
                        print(k)
                        print(getattr(stmt, k))
                        exit()
                args_string = ", ".join(args_string)
                subgoal_seq.append(construct+"("+args_string+")")
                subgoal_seq += self.extract_subgoals_from_subprograms(
                    stmt.body, inputs=subgoal_inputs, 
                    outputs=outputs, target=target,
                )
                if len(stmt.orelse) > 0: subgoal_seq.append("ORELSE")
                subgoal_seq += self.extract_subgoals_from_subprograms(
                    stmt.orelse, inputs=subgoal_inputs, 
                    outputs=outputs, target=target,
                )
                subgoal_seq.append("END"+construct)

            elif isinstance(stmt, (ast.Return, ast.Continue)):
                if not(first_goal): subgoal_seq.append("NEWLINE")
                else: first_goal = False
                subgoal_seq.append(type(stmt).__name__.upper())
            else: 
                print(stmt, ast.unparse(stmt))
                exit("Aborting! Decide how to handle this construct")

        return subgoal_seq

    def obfuscate_local_vars(self, subgoals: list, I0, O0):
        obf_mapping = {}
        for i in range(len(subgoals)):
            goal = subgoals[i]
            # print(goal)
            if not isinstance(goal, dict): continue
            for k in ["I", "O"]:
                for j, var in enumerate(goal[k]):
                    print(var, I0+O0)
                    if var in I0+O0: continue
                    obf_var = obf_mapping.get(var)
                    if obf_var is None:
                        obf_var = f"VAR{len(obf_mapping)}"
                        obf_mapping[var] = obf_var
                    # print(obf_var)
                    subgoals[i][k][j] = obf_var

        return subgoals, obf_mapping 
                    
    def serialize_subgoals(self, subgoals: List[str]):
        stream = []
        for subgoal in subgoals:
            if isinstance(subgoal, dict):
                stream.append(
                    f"[I] {' '.join(subgoal['I']).strip()}".strip() + " " + \
                    f"[O] {' '.join(subgoal['O']).strip()}".strip() + " " + \
                    # f"[T] {' '.join(subgoal['T']).strip()}".strip() + " " + \
                    f"[ops] {' '.join(subgoal['ops'])}".strip())
            elif isinstance(subgoal, str):
                stream.append(subgoal)

        return " [STEP] ".join(stream).strip()

    def add_type_annotations(self, code: str, input_annotations: Dict[str, str]) -> str:
        root = ast.parse(code)
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                for index in range(len(node.body)):
                    if isinstance(node.body[index], ast.Assign): break
                for var, var_type in list(input_annotations.items())[::-1]:
                    node.body.insert(index, ast.parse(f"{var}: {var_type} = None"))
                for index in range(len(node.body)):
                    if isinstance(node.body[index], ast.Return): break
                node.body.insert(index, ast.parse("reveal_type(movies['year'])"))
                break

        return ast.unparse(root)

    def __call__(self, code: str, remove_docstring: bool=True):
        if remove_docstring: code = remove_comments_and_docstrings(code)
        code = modify_complex_return_statements_in_code(code)
        I0 = extract_arguments_from_code(code)
        O0 = extract_returned_variables_from_code(code)
        self.init_io = (I0, O0)

        var_obf = VarNameObfuscater(input_list=I0, output_list=O0)
        obf_code = ast.unparse(var_obf.custom_visit(ast.parse(code)))
        # print(obf_code)
        # print(var_obf.inp_mapping, var_obf.out_mapping)
        obf_inputs = [var_obf.inp_mapping[i] for i in I0]
        obf_outputs = [var_obf.out_mapping[i] for i in O0]

        # TODO: try to finish this.
        # from model.poc.type_inference_aug import extract_docstring_from_function, extract_args_and_return_type_from_docstring
        # docstring = extract_docstring_from_function(code)
        # input_annotations, output_annotations = extract_args_and_return_type_from_docstring(docstring)
        # input_annotations = dict(input_annotations)
        # type_annotated_code = self.add_type_annotations(code, input_annotations)

        # print(type_annotated_code)
        subgoals = [{"I": obf_inputs, "O": obf_outputs, "ops": ["FUNCDEF"]}]

        funcdef = ast.parse(obf_code).body[0]
        assert isinstance(funcdef, ast.FunctionDef), "code is not formatted as a function"
        # subgoals += self.extract_subgoals_from_subprograms(
        #     funcdef.body, inputs=obf_inputs, 
        #     outputs=[], target=obf_outputs, 
        # )
        try: subgoals += self.extract_subgoals(funcdef.body)
        except NotImplementedError:
            print(obf_code)
            exit("Aborting! Decide how to handle this construct")
        var_mapping = {}
        var_mapping.update(var_obf.inp_mapping)
        var_mapping.update(var_obf.obf_mapping)
        var_mapping.update(var_obf.out_mapping)

        return subgoals, var_mapping, self.serialize_subgoals(subgoals)#, type_annotated_code
        # self.obfuscate_local_vars(subgoals, I0, O0)
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
    # chunk_tree, flat_chunk_list = extract_op_chunks(sample_code)
    # # print_chunks(chunks)
    # print(json.dumps(chunk_tree, indent=4))
    # print(json.dumps(flat_chunk_list, indent=4))
    nodecode2id, codecons2id, chunks = extract_plan_op_chunks_v2(sample_code)
    # chunks = [chunk.todict() for chunk in sorted([ChunkRepr(chunk) for chunk in chunks])]
    chunks = sort_and_remap_chunks(chunks)
    print(json.dumps(codecons2id, indent=4))
    # print(json.dumps(chunks, indent=4))
    def format_v(v):
        if isinstance(v, int):
            return f"\x1b[31;1m{v}\x1b[0m"
        elif isinstance(v, list):
            ret_out = []
            for subv in v:
                ret_out.append(format_v(subv))
            return "["+", ".join(ret_out)+"]"
        else: return v
    for chunk in chunks:
        chunk_args = [f'{k}={format_v(v)}' for k,v in chunk.items() if not(k.startswith('META'))]
        print(f"\x1b[31;1m{chunk['META_id']}\x1b[0m. \x1b[34;1m{chunk['META_chunktype']}\x1b[0m({', '.join(chunk_args)})")
        # print(f"\x1b[31;1m{chunk['META_id']}\x1b[0m")
        print(chunk['META_code'])