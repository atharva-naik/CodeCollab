# do type inference based augmentation
import re
import json
import typing
import inspect
import mypy.api
from model.poc.type_inference_class_method import get_return_type_of_class_function

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
    try:
        # Import the module dynamically
        module = __import__(module_name, fromlist=[function_name])
        # Get the function object
        function = getattr(module, function_name)
        # print("function:", function)
        # Get the function's signature
        signature = inspect.signature(function)
        # print("signature:", signature)
        # Get the return annotation
        return_annotation = signature.return_annotation

        return return_annotation
    # Handle import errors or attribute errors if the module or function is not found
    except (ImportError, AttributeError): return None

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
        break
    print(code)

# Example usage:
if __name__ == "__main__":
    # test_mypy_type_inference()
    # test_typehint_module_api_call_type_inference()
    test_typehint_with_FCDS_data()