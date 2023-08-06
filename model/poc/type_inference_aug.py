# do type inference based augmentation
import re
import ast
import json
import typing
import inspect
import mypy.api
import importlib
from typing import *
from python_graphs import control_flow, data_flow, cyclomatic_complexity
from model.poc.type_inference_class_method import get_return_type_of_class_function

def extract_docstring_from_function(code: str):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef):
            return ast.get_docstring(node)

# def extract_return_type(docstring):
#     # Find the return type pattern using regular expression
#     pattern = r'return\s*:\s*(.*?)\s*$'
#     match = re.search(pattern, docstring, re.MULTILINE)
    
#     if match:
#         return_type = match.group(1).strip()

#         # Create the dictionary as per the desired output format
#         output_dict = {}
#         for entry in return_type.split(','):
#             key, value = entry.split(':')
#             output_dict[key.strip()] = value.strip()

#         return return_type, output_dict
#     else:
#         return None, {}

def extract_args_and_return_type_from_docstring(docstring: str) -> tuple:
    """
    Extract the names/types of arguments and the return type from a function docstring.

    Args:
        docstring (str): The docstring of the function.

    Returns:
        tuple: A tuple containing a list of argument name/type pairs and the return type.
    """
    arg_type_pattern = r"(\w+)\s*\(([^)]+)\)\s*:"
    return_type_pattern = r"return\s*:\s*([^:]+)"

    arg_type_matches = re.findall(arg_type_pattern, docstring, re.MULTILINE)
    return_type_match = re.search(return_type_pattern, docstring, re.MULTILINE)

    args_and_types = arg_type_matches
    return_type = return_type_match.group(1).strip() if return_type_match else None

    return args_and_types, return_type
    # except ValueError: return args_and_types, return_type, {}
def extract_type_info_from_docstring(code) -> List[Tuple[str, str]]:
    """get docstring from the first function 
    definition and extract typing hints from it."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            return extract_args_and_return_type_from_docstring(docstring)

# def follow_args_in_code(code: str, argnames: List[str]):
#     for node in ast.walk(ast.parse(code)):
#         for argname in argnames:
#             if argname in ast.unparse(node):
#                 print(ast.unparse(node))

def extract_inferred_type(input_str: str) -> str:
    """
    Extract the inferred type from the multi-line input.

    Args:
        input_str (str): The multi-line input containing the inferred type.

    Returns:
        str: The extracted inferred type.
    (Partly Written by ChatGPT)
    """
    # Define the regular expression pattern to find the inferred type
    pattern = r'Revealed type is "([^"]*)"'
    # Search for the pattern in the input string
    match = re.search(pattern, input_str)
    # If a match is found, return the inferred type; otherwise, return None
    return match.group(1) if match else None

def get_return_type_of_module_function(module_name: str, function_name: str) -> typing.Optional[typing.Type]:
    """
    Get the return type annotation of a function defined in a module.

    Args:
        module_name (str): The name of the module containing the function.
        function_name (str): The name of the function.

    Returns:
        typing.Optional[typing.Type]: The return type annotation of the function, or None if not found.
    """
    # Import the module dynamically
    # module = importlib.import_module(module_name)
    module = __import__(module_name, fromlist=[function_name])
    # Get the function object
    function = getattr(module, function_name)
    assert inspect.isfunction(function), f"{module_name}.{function_name} has type {function}"
    # print("function:", function)
    # Get the function's signature
    signature = inspect.signature(function)
    print("signature:", signature)
    # Get the return annotation
    return_annotation = signature.return_annotation

    return return_annotation

def infer_type_of_variable_from_code(input_code: str, variable: str) -> str:
    """
    Perform type inference using mypy on the provided input code.
    
    Args:
        input_code (str): The Python code for which you want to perform type inference.
        
    Returns:
        str: The type inference results as a string.
    (Partly Written by ChatGPT)
    """
    # Prepare the arguments for mypy
    input_code = input_code + f"""
reveal_type({variable})"""
    mypy_args = ['--ignore-missing-imports', '-c', input_code]
    # Run mypy
    stdout, stderr, exit_status = mypy.api.run(mypy_args)
    # Combine the output and error messages
    # output = stdout + '\n' + stderr
    inferred_type = extract_inferred_type(stdout)
    # print(inferred_type)
    # Return the type inference results as a string
    return inferred_type, stdout, stderr

def test_mypy_type_inference():
    eg_code = """def add(a: int, b: int) -> int:
    return a + b

result = add(3, 5)
print(result)"""

    eg_code = """import pandas as pd
df = pd.DataFrame()"""
    print(infer_type_of_variable_from_code(eg_code, "df"))

def test_typehint_module_api_call_type_inference():
    module_name = "pandas"  # Replace with your module name
    function_name = "read_csv"  # Replace with your function name

    return_type_annotation = get_return_type_of_module_function(module_name, function_name)
    print(f"Return type annotation of '{function_name}': {return_type_annotation}")

    module_name = "torch.nn"
    class_name = "Module"
    function_name = "forward"
    return_type_annotation = get_return_type_of_class_function(module_name, class_name, function_name)

    print(f"Return type annotation of '{class_name}.{function_name}': {return_type_annotation}")

    module_name = "pandas"
    class_name = "DataFrame"
    function_name = "merge"
    return_type_annotation = get_return_type_of_class_function(module_name, class_name, function_name)

    print(f"Return type annotation of '{class_name}.{function_name}': {return_type_annotation}")

def test_typehint_with_FCDS_data():
    data = json.load(open("./data/FCDS/code_qa_submissions_and_chunks.json"))
    for intent, subs in data.items():
        for sub in subs:
            code = sub["code"]
            break
        # print(code)
        docstring = extract_docstring_from_function(code)
        arg_types, return_type = extract_type_info_from_docstring(code)
        print("-"*30)
        print(docstring)
        print("-"*30)
        print("\x1b[34;1mArgs:\x1b[0m")
        print(arg_types)
        print("\x1b[34;1m\nReturn:\x1b[0m")
        print(return_type)

        dfg = data_flow.get_dataflow_graph(code)
        cfg = control_flow.get_control_flow_graph(code)
        val = cyclomatic_complexity.cyclomatic_complexity(cfg)
        
        print(code)
        print(dfg)
        print(cfg)
        print(val)
        # print(output_dict)
        return

# Example usage:
if __name__ == "__main__":
    # test_mypy_type_inference()
    # test_typehint_module_api_call_type_inference()
    test_typehint_with_FCDS_data()