import logging

import numpy as np
import torch
from pyparsing import *
from ast import *
import copy
import ast
from typing import Union, Tuple, List, Any, Generator, Optional

from .parsers import language_to_parser
from ..utils import is_Text, is_str


def parse_content(content, parser):
    for i, c in enumerate(content):
        if not isinstance(c, int) and not c:
            content[i] = ""
    return to_ast_content(*content, parser=parser, is_tuple=True)


def to_ast_content(*args, parser="langtorch-f-string", is_tuple=False):
    """Reformats a wide array of construtor patterns into a unified AST-like format"""
    if len(args) == 0 or (len(args) == 1 and not args[0]):
        return ""
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
            return (arg[0], to_ast_content(arg[1], parser=parser)) if arg[0] else to_ast_content(arg[1], parser=parser)
        elif isinstance(arg, tuple) and len(arg) == 1:
            # CORRECT though should be avoided
            return arg[0]
        elif isinstance(arg, tuple) and len(arg) > 2:
            # Assume a tuple of length != 2 was supposed to be a list
            logging.debug(
                f"Tuples of length 2 represent (key, value) in Text objects. When parsing Text entry {arg} was a tuple of length {len(arg)},\nit was converted to a list and may lead to errors.")

            nonlocal args
            print("\n\n\n\n------",args)
            raise Exception("There is a tuple of length > 2 in a Text object. This indicates a bug in langtorch code.")

        # Not a named string
        if isinstance(arg, list) and len(arg) == 1:
            return simplify(arg[0], parser=parser)
        elif isinstance(arg, (list, np.ndarray, torch.Tensor)) and arg:
            return [simplify(element, parser=parser) for element in arg]
        elif is_Text(arg):
            if len(arg.items()) == 1:
                return to_ast_content(arg.items()[0], parser=parser)
            else:
                return to_ast_content(arg.items(), parser=parser)
        elif hasattr(arg, 'items'):
            return to_ast_content(list(arg.items()), parser=parser)
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
        return tuple((arg[0], arg[1]) if isinstance(arg, tuple) else ('', arg) for arg in content)


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


def is_valid_tree(entry, is_top_level=False):
    """
    Checks if an entry is a valid tree for a Text instance.

    """
    if is_Text(entry):
        entry = entry.items()
    elif isinstance(entry, tuple) and is_top_level:
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

import ast
from typing import Union, Any, List

class TextNode(ast.AST):
    def __init__(self, key: str, children: Union['TextNode', List['TextNode'], str], requires_grad=False):
        self.key = key
        self.children = children #if not isinstance(children, list) or len(children)==1 else children[0]
        self._requires_grad = requires_grad
        if isinstance(children, str):
            self._fields = ['key', 'children', '_requires_grad']
        else:
            self._fields = ['key', 'children']
    def __str__(self):
        q = lambda s: f"'{s}'"
        value = self.children if isinstance(self.children, list) else repr(self.children) if not isinstance(self.children, str) else q(self.children)
        if not self.key:
            return f"{value}"
        return f"({q(self.key)}, {value})"

    def __repr__(self):
        return str(self)

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self
    @property
    def requires_grad(self):
        if not isinstance(self.children, str):
            self._requires_grad = all(child.requires_grad for child in self.list_children())
        return getattr(self, '_requires_grad', False)
    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        self._requires_grad = requires_grad
        if not isinstance(self.children, str):
            if isinstance(self.children, list):
                for child in self.children:
                    child.requires_grad_(requires_grad)
            else:
                self.children.requires_grad_(requires_grad)
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_node = TextNode(
            key=copy.deepcopy(self.key, memo),
            children=copy.deepcopy(self.children, memo),
            requires_grad=self._requires_grad
        )
        memo[id(self)] = new_node
        return new_node


    def list_children(self):
        if isinstance(self.children, list):
            return self.children
        else:
            return [self.children]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the _fields attribute as it's not necessary for reconstruction
        state.pop('_fields', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct the _fields attribute
        if isinstance(self.children, str):
            self._fields = ['key', 'children', '_requires_grad']
        else:
            self._fields = ['key', 'children']


class TextAST:
    def __init__(self, tree: Union[ast.AST, 'Text']):
        self.tree = tree if isinstance(tree, ast.AST) else self.to_ast(tree)

    def to_ast(self, content) -> ast.Module:
        def build_ast_node(node: Union[str, tuple, List]) -> Union[TextNode, List[TextNode]]:
            if isinstance(node, str):
                return TextNode(key='', children=node)
            elif isinstance(node, tuple) and len(node) == 2:
                key, value = node
                if isinstance(value, (str, tuple)):
                    return TextNode(key=key, children=build_ast_node(value))
                elif isinstance(value, list):
                    return TextNode(key=key, children=[build_ast_node(child) for child in value])
            elif isinstance(node, (list, tuple)):
                return [build_ast_node(child) for child in node]
            raise ValueError(f"Unexpected node type: {type(node)}")

        if is_Text(content):
            content = content.items()

        return ast.Module(body=[ast.Expr(value=build_ast_node(content))], type_ignores=[])

    @staticmethod
    def _ast_to_python(node: Union[TextNode, List[TextNode]], include_flags=False) -> Any:
        if isinstance(node, TextNode):
            if isinstance(node.children, str):
                terminal= f"<REQUIRED_GRAD>{node.children}</REQUIRED_GRAD>" if include_flags and node.requires_grad else node.children
                return (node.key, terminal) if node.key else terminal
            elif isinstance(node.children, list):
                return (node.key, [TextAST._ast_to_python(child, include_flags) for child in node.children])
            else:
                return (node.key, TextAST._ast_to_python(node.children, include_flags))
        elif isinstance(node, list):
            return [TextAST._ast_to_python(child, include_flags) for child in node]
        else:
            raise ValueError(f"Unexpected AST node type: {type(node)}")

    def to_python(self, include_flags=False) -> Any:
        # print(TextAST._ast_to_python(self.tree.body[0].value, include_flags))
        return TextAST._ast_to_python(self.tree.body[0].value, include_flags)


    def replace_subtree(self, index: Union[int, tuple], new_subtree: Union['Text', 'TextAST']) -> 'TextAST':
        from langtorch import Text

        def replace_node(node: Union[TextNode, List[TextNode]], idx: Union[int, tuple], new_node: TextNode) -> None:
            if isinstance(idx, int):
                idx = (idx,)
            current = node
            for i in idx[:-1]:
                if isinstance(current, list):
                    current = current[i]
                elif isinstance(current, TextNode):
                    if isinstance(current.children, list):
                        current = current.children[i]
                    else:
                        current = current.children
                else:
                    raise IndexError(f"Cannot index {type(current)} with integer")
            if isinstance(current, list):
                current[idx[-1]] = new_node
            elif isinstance(current, TextNode):
                if isinstance(current.children, list):
                    current.children[idx[-1]] = new_node
                else:
                    current.children = new_node

        if is_Text(new_subtree):
            new_subtree = new_subtree.tree
        new_ast = new_subtree.tree.body[0].value
        if isinstance(new_ast, list) and len(new_ast) == 1:
            new_ast = new_ast[0]

        replace_node(self.tree.body[0].value, index, new_ast)
        return self

    def __str__(self):
        return ast.dump(self.tree, indent=2)

    def to_text(self):
        from langtorch import Text
        return Text(self._ast_to_python(self.list_children()))

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self
    @property
    def requires_grad(self):
        # if not isinstance(self.children, str):
        #     self._requires_grad = all(child.requires_grad for child in self.children)
        return getattr(self, '_requires_grad', False)
    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        self._requires_grad = requires_grad
        for child in self.list_children():
            child.requires_grad_(requires_grad)

    def list_children(self):
        child = self.tree.body[0].value
        return child if isinstance(child, list) else [child]

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        new_ast = TextAST.__new__(TextAST)
        memo[id(self)] = new_ast
        new_ast.tree = copy.deepcopy(self.tree, memo)
        return new_ast

    def __getitem__(self, item):
        return self.get_by_index(item)

    def __getstate__(self):
        return {'tree': self.tree}

    def __setstate__(self, state):
        self.tree = state['tree']

    def traverse(self, node=None, path=None, index=None) -> Generator[Tuple[TextNode, list, list], None, None]:
        if node is None:
            node = self.list_children()
        if path is None:
            path = []
        if index is None:
            index = []

        def process_node(current_node, current_path, current_index):
            yield current_node, current_path, current_index

            if isinstance(current_node, TextNode):
                if isinstance(current_node.children, list):
                    for i, child in enumerate(current_node.children):
                        yield from process_node(child, current_path + [current_node], current_index + [i])
                elif not isinstance(current_node.children, str):
                    yield from process_node(current_node.children, current_path + [current_node], current_index)
            elif isinstance(current_node, list):
                for i, item in enumerate(current_node):
                    yield from process_node(item, current_path, current_index + [i])
            elif not isinstance(current_node, str):
                raise ValueError(f"Unexpected node type: {type(current_node)}")

        yield from process_node(node, path, index)

    def build_path(self, path):
        if not path:
            return None
        current = path[-1]
        for parent in reversed(path[:-1]):
            new_parent = TextNode(
                key=parent.key,
                children=current,
                requires_grad=parent.requires_grad
            )
            current = new_parent
        return current

    def __iter__(self):
        for node, path, _ in self.traverse():
            if isinstance(node, TextNode) and isinstance(node.children, str):
                yield self.build_path(path + [node])

    def enumerate(self):
        for node, path, index in self.traverse():
            if isinstance(node, TextNode) and isinstance(node.children, str):
                yield tuple(index), self.build_path(path + [node])

    def get_by_index(self, index: Union[int, tuple]) -> Optional['TextAST']:
        if isinstance(index, int):
            index = (index,)

        for node, _, current_index in self.traverse():
            if tuple(current_index) == index:
                return TextAST(ast.Module(body=[ast.Expr(value=node)], type_ignores=[]))

        raise IndexError(f"Index {index} not found")

