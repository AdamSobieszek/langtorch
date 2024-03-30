import logging

import numpy as np
import torch
from pyparsing import *

from .parsers import language_to_parser
from ..utils import is_Text, is_str


def parse_content(content, parser):
    for i,c in enumerate(content):
        if not isinstance(c, int) and not c:
            content[i] = ""
    return to_ast(*content, parser=parser, is_tuple=True)


def to_ast(*args, parser="langtorch-f-string", is_tuple=False):
    """Reformats a wide array of construtor patterns into a unified AST-like format"""
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    if len(args) == 1 and ((isinstance(args[0], tuple) and len(args[0]) > 2) or isinstance(args[0], list)):
        # List or tuple of strings / named strings
        args = args[0]
    elif isinstance(args[0], dict):
        # Dictionary of named strings
        args = args[0].items()
    elif is_Text(args[0]) and len(args) == 1:
        # Passing cls instance to itself
        args = args[0].content
    elif all([isinstance(arg, str) for arg in args]):
        if parser:
            try:
                result = []
                for arg in args:
                    arg = parse_string(arg, parser=parser)
                    result += arg
                args = result
            except ParseException as E:
                print(f"Last parsed string: {arg}")
                raise ParseException(str(E) + "\nYou may want to disable string parsing, with parse = False")
        else:
            pass
    if any([isinstance(arg, torch.Tensor) for arg in args]):
        raise ValueError(
            "You cannot initialise Text from a TextTensor. Use tensors.item() or otherwise transform the tensors to a string, list or dictionary.")

    def simplify(arg, parser=False):
        if isinstance(arg, tuple) and len(arg) == 2 and not (isinstance(arg[0], str) and isinstance(arg[1], str)):
            # Fix tuple types
            return (str(arg[0]), simplify(arg[1])) if str(arg[0]) else simplify(arg[1])
        elif isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], str) and isinstance(arg[1], str):
            # CORRECT: (key, value) tuple
            return (arg[0], to_ast(arg[1], parser=parser)) if arg[0] else to_ast(arg[1], parser=parser)
        elif isinstance(arg, tuple) and len(arg) == 1:
            # CORRECT though should be avoided
            return arg[0]
        elif isinstance(arg, tuple) and len(arg) > 2:
            # Assume a tuple of length != 2 was supposed to be a list
            logging.debug(
                f"Tuples of length 2 represent (key, value) in Text objects. When parsing a Text entry was a tuple of length {len(arg)},\nit was converted to a list and may lead to errors.")
            arg = list(arg)

        # Not a named string
        if isinstance(arg, list) and len(arg) == 1:
            return simplify(arg[0])
        elif isinstance(arg, (list, np.ndarray, torch.Tensor)):
            return [simplify(element, parser=parser) for element in arg]
        elif is_Text(arg):
            if len(arg.items()) == 1:
                return to_ast(arg.items()[0], parser=parser)
            else:
                return to_ast(arg.items(), parser=parser)
        elif hasattr(arg, 'items'):
            return to_ast(list(arg.items()), parser=parser)
        elif isinstance(arg, str):
            # CORRECT
            return arg
        else:  # Cast to string
            return str(arg)
        # Maybe consider: raise ParseException(f"Could not parse {arg} of type {type(arg).__name__}")

    content = [simplify(arg, parser=parser) for arg in args]
    if not is_tuple:  # Recursive case: In these cases we are returning a node or tree with a single root node
        return content[0] if isinstance(content, list) and len(content) == 1 else content
    else:  # Base case: In these cases we are returning a tuple of nodes for the ._content attribute
        def check_for_lists(tree):  # A basic check for lists in the tree
            if isinstance(tree, list):
                # Cast 0-length lists to empty strings and check for 1-length lists
                if len(tree) == 1:
                    raise ValueError(f"single-element list {tree}")
                else:
                    return [check_for_lists(element) for element in tree] if tree else ""
            elif isinstance(tree, tuple) and len(tree) == 2:
                key, value = tree
                return key, check_for_lists(value)
            return tree

        content = tuple(check_for_lists(arg) for arg in content)

        return content


def parse_string(arg, parser):
    if not parser:
        return arg
    else:
        return language_to_parser[parser](arg)


def is_terminal_node(entry):
    if is_str(entry):
        return True
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], str):
        return True
    return False


def is_valid_tree(entry, is_tuple=False):
    """
    Checks if an entry is a valid tree for a Text instance.

    """
    if is_Text(entry):
        entry = entry.items()
    elif isinstance(entry, tuple) and is_tuple:
        if len(entry) == 0:
            return True
        entry = list(entry) if len(entry) > 1 else entry[0]
    # Base cases
    if is_terminal_node(entry):
        return True

    # Recursive cases
    if isinstance(entry, tuple) and len(entry) != 2:
        return False
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) and (
            isinstance(entry[1], list) or isinstance(entry[1], tuple)):
        return all(is_valid_tree(child) for child in entry[1])
    if isinstance(entry, list) and len(entry) > 1:
        return all(is_valid_tree(child) for child in entry)
    if isinstance(entry, list) and len(entry) <= 1:
        return False  # Single-element lists are not valid

    # If none of the above cases match, it's not valid
    return False
