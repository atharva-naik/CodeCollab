import inspect
import typing
import random
import importlib

def get_return_type_of_class_function(module_name: str, class_name: str, function_name: str) -> typing.Optional[typing.Type]:
    """
    Get the return type annotation of a member function of a class

    Args:
        module_name (str): module containing the class.
        class_name (str): The name of the class.
        function_name (str): The name of the member function.

    Returns:
        typing.Optional[typing.Type]: The return type annotation of the member function (throws error if not found)
    """
    # Dynamically import the module containing the class
    module = importlib.import_module(module_name)
    # Get the class
    class_ = getattr(module, class_name)
    # Get the member function
    function = getattr(class_, function_name)
    # Get the function's signature
    signature = inspect.signature(function)
    print(signature)
    # Get the return annotation
    return_annotation = signature.return_annotation

    return return_annotation

# Example usage:
if __name__ == "__main__":
    class Person:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

    class Point:
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

    class Rectangle:
        def __init__(self, top_left: Point, bottom_right: Point):
            self.top_left = top_left
            self.bottom_right = bottom_right

        def get_bottom_right(self) -> Point:
            pass

    module_name = "torch.utils.data"
    class_name = "DataLoader"
    function_name = "__init__"

    return_type_annotation = get_return_type_of_class_function(module_name, class_name, function_name)
    print(f"Return type annotation of '{function_name}': {return_type_annotation}")