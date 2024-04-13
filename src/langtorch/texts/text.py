import json
import logging
import re
from typing import List, Any, Tuple, Union
import inspect
import os

from pyparsing import *

from ..grammars import formatters
from ..grammars import parsers
from ..grammars import text_content_ast as ast
from ..utils import is_TextTensor, is_Text


class Text(str):
    """
    The Text class represents structured textual data within LangTorch, allowing for complex manipulations
    and operations on text similar to working with ordered dictionaries but with enhanced string manipulation capabilities.

    Attributes:
        content: The core data of a Text instance, holding a sequence of "named strings" or key-value pairs. This attribute
                 allows Text to support operations like concatenation, formatting, and structured manipulation.
        language: Specifies the language for parsing and formatting the textual content. Default is "str", but can be set
                  to other languages to support various textual operations and transformations.
        allowed_keys: Optionally specifies a list of keys that are permitted within the Text instance. Attempts to use
                      keys not in this list will raise a ValueError, enforcing a schema on the textual data.

    Initialization:
        Text instances can be initialized in multiple ways, including direct strings, tuples representing key-value pairs,
        dictionaries, or even from other Text instances. This flexibility allows developers to easily structure their textual
        data as needed for their applications.

    Notes:
        - Text instances support advanced text manipulation operations, including formatting via multiplication with other
          Text instances or dictionaries, concatenation with other Text instances or strings, and even splitting into
          TextTensor for further processing.
        - The `iloc` and `loc` properties provide powerful mechanisms for accessing and manipulating sub-elements of a Text
          instance based on index or key, respectively, facilitating easy modifications and queries.

    Warnings:
        - Be cautious when using keys not specified in `allowed_keys` as it will raise a ValueError.
        - Incorrect use of formatting operations or accessing non-existent keys/indexes may lead to unexpected outcomes.

    Examples:
        Initializing a Text instance:
        >>> text = Text("Hello, {name}!", name="World")
        >>> print(text)
        "Hello, World!"

        Concatenating Text instances:
        >>> greeting = Text("Hello")
        >>> target = Text(name="World")
        >>> combined = greeting + ", " + target
        >>> print(combined)
        "Hello, World"

        Formatting Text instances:
        >>> template = Text("Dear {title} {last_name},")
        >>> formatted = template * {"title": "Mr.", "last_name": "Doe"}
        >>> print(formatted)
        "Dear Mr. Doe,"

        Accessing attributes:
        >>> text = Text("First", "Second", key1="Value1", key2="Value2")
        >>> print(text.keys())
        ['key1', 'key2']
        >>> print(text.items())
        [('key1', 'Value1'), ('key2', 'Value2')]
        >>> print(text.iloc[0])
        "First"
        >>> print(text.loc['key1'])
        "Value1"

    """

    parsers = parsers.language_to_parser
    formatters = formatters.language_to_formatter
    language = "str"
    allowed_keys = None

    def __new__(cls, *substrings, parse: Union[str, bool] = "langtorch-f-string", language="str", **named_substrings):
        """
        Construct a new Text instance. Allows for various input formats.

        Args:
            *substrings (Union[str,Tuple[str, str], List[str]): Flexible input data. Can be a parsable string, string sequences, key-value pairs, dicts...
                                    If None is passed, it will be replaced with a Text instance with empty content.
            parse (Union[bool, str], optional): Disable or specify a parsing langauge the input content is written.
                                        The default behavior is splits strings with an f-string-like syntax.
                                        You can pass a name of a markup language to parse it with pandoc.
                                        Set to False to disable parsing.
            language (str, optional): The language that the content should be translated to when casting to string.
            **named_substrings (str): Additional named textual data entries.

        Returns:
            (Text): A structured textual instance.

        Raises:
            ParseException: If automatic parsing fails. Consider disabling parsing if this occurs.
            ValueError: When an unsupported input format is provided e.g. a TextTensor is passed.
        """
        if len(substrings) == 0 or (len(substrings) == 1 and substrings[0] is None):
            instance = super().__new__(cls, "")
            instance._content = tuple()
            return instance
        content = [c if c is not None else cls.identity for c in (list(substrings) + list(named_substrings.items()))]
        # cast TextTensors to strings
        for i in range(len(content)):
            if is_TextTensor(content[i]):
                content[i] = content[i].sum().item()
                assert isinstance(content[i], str) or (0 < len(content[i]) <= 2)
        # returns the final content tuple
        parser = "langtorch-f-string" if parse is True else parse
        content = cls._parse_content(content, parser=parser)

        assert cls._is_valid_tree(content, is_tuple=True), f"Creating Text with an invalid content tree: {content}"

        instance = super().__new__(cls, cls.str_formatter(content))
        instance._content = content
        instance.language = language
        if instance.allowed_keys is not None and any([k not in instance.allowed_keys for k in instance.keys()]):
            raise ValueError(
                f"Invalid key found in {instance.keys()}. For class {cls.__name__} only {instance.allowed_keys} keys are allowed.")
        return instance

    _is_valid_tree = ast.is_valid_tree
    _is_terminal_node = ast.is_terminal_node
    _to_ast = ast.to_ast
    _parse_content = ast.parse_content

    @classmethod
    def str_formatter(cls, instance, language="str") -> str:
        """
        Formats the human-readable string of a Text instance. Subclasses of Text can reimplement this method!

        Args:
            instance (Text): An instance of the Text class.

        Returns:
            (str): A string representation of the instance.
        """
        return cls.formatters[language](instance)

    def __str__(self):
        s = self.__class__.str_formatter(self, self.language)
        return s if s else "\u200B"  # Zero width space to fix errors with printing arrays of empty Texts

    @classmethod
    def from_messages(cls, *messages, **kwargs):
        """Text from a list of dicts with keys 'role' and 'content'"""
        if len(messages) == 0:
            return cls()
        if all(isinstance(m, list) for m in messages):
            return [cls.from_messages(*m, **kwargs) for m in messages]

        text = []
        for m in messages:
            if "message" in m and "content" in m["message"]:
                m = m["message"]
            assert "role" in m and "content" in m, f"Each message must have a 'role' and 'content' key. Got {m}"
            if m["content"]:
                text.append((m["role"], m["content"]))
            if "tool_calls" in m and m["tool_calls"]:
                for call in m["tool_calls"]:
                    text.append((m["role"], ("tool_call", json.dumps(call, ensure_ascii=False))))
                    kwargs["parse"] = False
        return cls(*text, **kwargs)

    @classmethod
    def from_api_responses(cls, *messages, **kwargs):
        """Text from a list of dicts of OpenAI style API responses"""
        if len(messages) == 0:
            return cls()
        if not isinstance(messages[0], dict):
            if len(messages) == 1 and isinstance(messages[0][0], dict):
                messages = messages[0]
            else:
                raise ValueError(f"Expected a list of dict type api responses, got {messages}")

        text = cls.from_messages([m["choices"] for m in messages], **kwargs)
        return cls(*text, **kwargs)

    def keyed_print(self):
        """
        Prints the Text instance with keys aligned over their corresponding values.
        """
        # Extract the key-value pairs from the _content attribute
        key_value_pairs = [pair if isinstance(pair, tuple) else ('', pair) for pair in self.items()]

        # Find the longest key and value for formatting purposes
        longest_key = max(len(key) for key, _ in key_value_pairs)
        longest_value = max(len(value) for _, value in key_value_pairs)

        # Create the top and bottom lines for the keys and values
        top_line = ""
        bottom_line = ""
        for key, value in key_value_pairs:
            # Calculate padding to center the key
            total_padding = longest_key - len(key)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding

            # Add the formatted key and value to the respective lines
            top_line += f"{' ' * left_padding}{key}{' ' * right_padding}|"
            bottom_line += f"{value.ljust(longest_value)}|"

        # Remove the trailing '|' character
        top_line = top_line.rstrip('|')
        bottom_line = bottom_line.rstrip('|')

        # Print the formatted lines
        print(top_line)
        print(bottom_line)

    def __repr__(self):
        return self.__str__()

    def to_tensor(self):
        from langtorch import TextTensor
        return TextTensor(self.content)

    @property
    def identity(self):
        return self.__class__()

    @property
    def content(self):
        return [self.__class__(m, parse=False) for m in self._content]

    @content.setter
    def content(self, content):
        if isinstance(content, tuple) and len(content) == 2 and (
                isinstance(content[0], str) and isinstance(content[1], str)):
            AttributeError(
                f"When setting the .content attribute, passing a tuple of two strings is ambiguous. Pass a list with a tuple [(key, value)] or a list of strings [value, value]."
            )

        if isinstance(content, list):
            content = tuple(content)
        assert isinstance(content, tuple)
        self._content = self._to_ast(content, parser=False, is_tuple=True)

    def items(self):
        """
        Retrieves key-value pairs from the Text object, allowing for structured data extraction
        and further processing.

        Returns:
            (List[Tuple[str, Union[str, Tuple[...]]]]): A list of key-value pairs representing the Text's content.
        """
        return [(arg[0], arg[1]) if isinstance(arg, tuple) else ('', arg) for arg in self._content]
        # return [(arg[0], str(self.__class__(arg[1])))) if isinstance(arg, tuple) else ('', str(self.__class__(arg))) for arg in self._content]

    def keys(self):
        return [s[0] for s in self.items()]

    def values(self):
        return [s[1] for s in self.items()]

    def set_key(self, key, inplace=False):
        """
        Override keys for the textual entries, used for restructuring the content.
        Useful for substituting the key right before passing TextTensor to a Module.

        Args:
            key (Union[Text, str, List[str]]): The new key or keys to apply
            inplace bool: .

        Returns:
            (Text): A new Text instance with updated keys.
        """
        # The use of Text.str_formatter(t) instead of str(t) here and elsewhere is for subclasses of Text to reimplement __str__
        if isinstance(key, Text):
            key = key.values()
            if len(key) == 1:
                key = key[0]

        if isinstance(key, list):
            assert len(key) == len(
                self.values()), f"Number of keys ({len(key)}) must match number of values ({len(self.values())})"
            content = tuple((k, v) for k, v in zip(key, self.values()))
        elif isinstance(key, str):
            content = ((key, Text.str_formatter(self)),)
        if inplace:
            self.content = content
            return self
        else:
            return self.__class__(*content, parse=False)

    def set_key_(self, keys):
        self.set_key(keys, inplace=True)

    def add_key(self, key, inplace=False):
        """
        Add a top-level  key, placing the items of the original as a value under the new key.
        Useful for working with nested keys like in Chat prompts.

        Args:
            key (Union[Text, str, List[str]]): The new key to add
            inplace bool: .

        Returns:
            (Text): A new Text instance with updated keys.
        """
        # The use of Text.str_formatter(t) instead of str(t) here and elsewhere is for subclasses of Text to reimplement __str__
        if isinstance(key, Text):
            key = key.values()
            if len(key) == 1:
                key = key[0]

        if isinstance(key, list) and len(key) > 1:
            assert len(key) == len(
                self.content), f"Number of keys ({len(key)}) must match number of entries ({len(self.content)})"
            content = tuple((k, v) for k, v in zip(key, self.content))
        elif isinstance(key, str):
            content = ((key, self._to_ast(self.items(), parser=False)),)

        if inplace:
            self.content = content
            return self
        else:
            return self.__class__(*content, parse=False)

    def add_key_(self, keys):
        return self.add_key(keys, inplace=True)

    @property
    def iloc(self):
        """
        Index-based indexing. You can access entries with nested keys using dot notation, e.g. .loc["key1.key2"]
        """

        class IlocIndexer:
            def __init__(self, text_instance):
                self.text_instance = text_instance

            def __getitem__(self, index):
                items = self.text_instance.items()
                if isinstance(index, int):
                    # Support negative indices
                    if index < 0:
                        index += len(items)
                    try:
                        item = items[index]
                        return self.text_instance.__class__(item, parse=False)
                    except IndexError:
                        raise IndexError(f"Index {index} out of range")
                elif isinstance(index, slice):
                    sliced_items = items[index]
                    return self.text_instance.__class__(sliced_items, parse=False)
                elif isinstance(index, list):
                    # Ensure all elements are integers
                    if not all(isinstance(i, int) for i in index):
                        raise IndexError("Index list must contain only integers")
                    selected_items = [items[i] for i in index]
                    return self.text_instance.__class__(selected_items, parse=False)
                else:
                    raise TypeError("Index must be an int, slice, or list of ints")

            def __setitem__(self, index, value):
                # Convert single value to tuple for consistency
                if not isinstance(value, tuple) or len(value) == 2:
                    value = (value,)

                # Handle single index assignment
                if isinstance(index, int):
                    if index < 0:
                        index += len(self.text_instance._content)
                    self.text_instance._content = (
                            self.text_instance._content[:index] + value + self.text_instance._content[index + 1:]
                    )

                # Handle slice assignment
                elif isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = len(self.text_instance._content) if index.stop is None else index.stop

                    self.text_instance._content = (
                            self.text_instance._content[:start] +
                            value +
                            self.text_instance._content[stop:]
                    )
                else:
                    raise IndexError("Index must be an integer or a slice for setting items.")

        return IlocIndexer(self)

    @property
    def loc(self):
        """
        Key-based indexing. You can access entries with nested keys using dot notation, e.g. .loc["key1.key2"]
        """

        class LocIndexer:
            def __init__(self, text_instance):
                self.text_instance = text_instance
                self.ttype = text_instance.__class__

            def __getitem__(self, key):
                """
                Get item(s) from the TextTensor based on the key.
                Supports nested access using dot notation.
                :param key: A string or list of strings representing keys.
                :return: A TextTensor instance with the requested items.
                """
                if not isinstance(key, (str, list)):
                    raise TypeError(f"Key must be a string or list, not {type(key)}")

                def get_subitems(items, keys):
                    """
                    Helper function to recursively get items, given a list of nested keys.
                    :param items: The list of items to search through.
                    :param keys: The list of keys to match.
                    :return: A list of matching sub-items.
                    """
                    if not keys:
                        return items
                    next_key = keys.pop(0)
                    sub_items = [(k, get_subitems(v if isinstance(v, list) else [v], keys.copy())) for k, v in items if
                                 k == next_key]
                    return sub_items
                    if False:
                        # Optional version with flatten: i think it is better to keep the nested structure
                        nested_items = [[v] if isinstance(v, str) else v for _, v in sub_items]
                        return [item for sublist in nested_items for item in get_subitems(sublist, keys.copy())]

                def generate_nested_keys(item, keys):
                    """
                    puts each item onto a nested tuple of keys
                    """
                    if len(keys) == 0:
                        return item
                    return generate_nested_keys((keys.pop(-1), item), keys)

                items = self.text_instance.items()
                if isinstance(key, str):
                    keys = key.split('.') if '.' in key else [key]
                    return self.ttype([generate_nested_keys(v, keys) for v in get_subitems(items, keys)],
                                      parse=False)
                else:  # key is a list
                    return self.ttype([(k, v) for k, v in items if k in key], parse=False)

            def __setitem__(self, key, value):
                """
                Set or update item(s) in the TextTensor.
                :param key: A string or list of strings representing keys.
                :param value: The value to set or update.
                """
                if not isinstance(key, (str, list)):
                    raise TypeError(f"Key must be a string or list, not {type(key)}")

                def change_content_tuple(tup, keys, value):
                    """
                    Helper function to set items in a tuple.
                    :param tup: The tuple to update.
                    :param keys: The list of keys for updating.
                    :param value: The value to set.
                    :return: A tuple with updated items.
                    """
                    if isinstance(keys, str):
                        keys = [keys]
                    if not isinstance(value, Text):
                        value = Text(value)

                    new_tup = []
                    relevant_items = [(k, v) for k, v in tup if k in keys]
                    # Editing case
                    if len(relevant_items) <= len(value.items()):
                        replacement_items = list(value.items()[:len(relevant_items)])
                        append_items = list(value.items()[len(relevant_items):])
                    else:
                        replacement_items = [value.items()] * len(relevant_items)
                        append_items = []
                    relevant_i = 0
                    for item in tup:
                        k, v = item if isinstance(item, tuple) else ('', item)
                        if k in keys:
                            new_v = replacement_items[relevant_i]
                            if not isinstance(new_v, list):
                                new_v = [new_v]
                            new_v += append_items
                            if len(new_v) == 1:
                                new_v = new_v[0]
                            new_item = (k, new_v)
                            new_tup.append(new_item)
                            relevant_i += 1
                        else:
                            new_tup.append(item)
                    return tuple(new_tup)

                self.text_instance.content = change_content_tuple(self.text_instance._content, key, value)

        return LocIndexer(self)

    def split(self, sep=" ", mode="auto"):
        modes = ["auto", "words", "sentances", "paragraphs"]  # TODO maybe use spliters from other packages
        assert mode in modes, f"Mode must be one of {modes}"
        if mode != "auto":
            raise NotImplementedError("Other modes are not yet implemented")
        from langtorch import TextTensor
        return TextTensor(str(self).split() if sep == "" else str(self).split(sep), parse=False)


    def __getitem__(self, index):
        if isinstance(index, str):
            return self.loc[index]
        return self.iloc[index]

    def __or__(self, other):
        return

    def __len__(self):
        return len(str(self.content))

    def __iter__(self):
        for s in self.content:
            yield s

    # def _handle_other(self, other, method):
    #     if isinstance(other, str) and not isinstance(other, Text):
    #         return other
    #     elif isinstance(other, Text):
    #         return other.content
    #     elif is_TextTensor(other):
    #         if len(other.flat) == 1:
    #             return self.__class__(*self.content, *other.item().content, parse=False)
    #         else:
    #             return other.__class__([getattr(self, method)(t) for t in other.flat], ttype=self.__class__, parse=False)
    #     else:
    #         return other

    def __add__(self, other):
        if isinstance(other, str) and not isinstance(other, Text):
            return self.__class__(*self.content, other, parse=False)
        elif isinstance(other, Text):
            return self.__class__(*self.content, *other.content, parse=False)
        elif is_TextTensor(other):
            if len(other.flat) == 1:
                return self.__class__(*self.content, *other.item().content, parse=False)
            else:
                return other.__class__([self + t for t in other.flat], ttype=self.__class__, parse=False)
        else:
            try:
                return self.__class__(*self.content, other, parse=False)
            except ValueError as e:
                raise ValueError(f"Cannot add other={other} to Text. Failed to create a Text instance from other.") from e

    def __mul__(self, other):
        if is_TextTensor(other):
            if len(other.flat) == 1:
                other =  other.item()
            else:
                return other.__class__([self * t for t in other.flat], ttype=self.__class__, parse=False)
        elif not isinstance(other, Text):
            try:
                other = Text(other)
            except ParseException:
                other = Text(other, parse=False)

        def flatten_keys(parent_key, value, sep='.'):
            if isinstance(value, tuple):
                if not parent_key:
                    return flatten_keys(value[0], value[1])
                return flatten_keys(f"{parent_key}{sep + value[0] if value[0] else ''}", value[1])
            else:
                return (parent_key, value)

        def match_mul_rule(k, v, k_, v_, i, result, formatted_indices, indices_to_delete):
            k, v = flatten_keys(k, v)

            if isinstance(v, list):
                subresult = v  # still a refence to the original
                for subi, subv in enumerate(v):
                    subi = (i, subi) if isinstance(i, int) else i + (subi,)
                    if subi not in formatted_indices:
                        subk, subv = flatten_keys("", subv)
                        if match_mul_rule(subk, subv, k_, v_, subi, subresult, formatted_indices, indices_to_delete):
                            result[i] = (k, subresult)
                            return True
            elif i not in formatted_indices:
                ind = (i if isinstance(i, int) else i[-1])
                if (k, v) == (v_, k_):
                    formatted_indices.append(i)
                    indices_to_delete.append(i)
                    return True
                elif v == k_:
                    result[ind] = (k, v_)
                    formatted_indices.append(i)
                    return True
            return False

        def handle_positional_args(k_, v_, i, result, formatted_indices, indices_to_delete, positional_j):
            for j, (k, v) in enumerate(content):
                if v == str(positional_j):
                    result[j] = (k, v_)
                    positional_j += 1
                    formatted_indices.append(j)
                    return positional_j
            for j, (k, v) in enumerate(content):
                if v == "":
                    result[j] = (k, v_)
                    formatted_indices.append(j)
                    if v_ == k:
                        indices_to_delete.append(j)
                    return positional_j
            result.append((k_, v_))
            return positional_j

        content = self.items()
        result = content[:]
        formatted_indices = []
        indices_to_delete = []
        positional_j = 0

        for i, (k, v) in enumerate(content):
            if v == "*":
                if k == "":
                    result = result[:i] + list(other.items()) + result[i + 1:]
                else:
                    result[i] = (k, other.items())
                return self.__class__(*result, parse=False)

        for k_, v_ in other.items():
            if v_ == "*":
                result = [(k_, [item for i, item in enumerate(result) if i not in indices_to_delete])]
                indices_to_delete = []
                formatted_indices = []
            elif k_ == "":
                positional_j = handle_positional_args(k_, v_, i, result, formatted_indices, indices_to_delete,
                                                      positional_j)
            else:  # Key substitution
                k_, v_ = flatten_keys(k_, v_)
                for i, (k, v) in enumerate(content):
                    if match_mul_rule(k, v, k_, v_, i, result, formatted_indices, indices_to_delete):
                        break
                else:
                    result.append((k_, v_))

        def del_ind(subresult, index):
            if isinstance(subresult, tuple):
                return (subresult[0], del_ind(subresult[1], index))
            if isinstance(index, int):
                return subresult[:index] + subresult[index + 1:]
            else:
                sublist = del_ind(subresult[index[0]], index[1:] if len(index[1:]) > 1 else index[1])
                return subresult[:index[0]] + [sublist] + subresult[index[0] + 1:]

        # Sort the indices in reverse order based on their depth and values
        def sort_key(index):
            if isinstance(index, int):
                return (0, index)
            else:
                return (len(index), index)

        indices_to_delete.sort(key=sort_key, reverse=True)

        for i in indices_to_delete:
            result = del_ind(result, i)

        return self.__class__(*result, parse=False)

    def format(self, *args, **kwargs):
        other = Text(*args, parse=False) + Text(kwargs, parse=False)
        available_values = self.values()
        entries_with_corresponding_values = []
        for i, k in enumerate(other.keys()):
            if k in available_values:
                entries_with_corresponding_values.append(i)
                available_values.pop(available_values.index(k))
        return self.__mul__(other, strict=True)

    def inv(self):
        return self.__class__(*[(v, k) for k, v in self.items()], parse=False)

    def __pow__(self, power):
        if power == -1:
            return self.inv()
        else:
            raise ValueError("Can only use power -1")

    def method_apply(self, method: str, *args, to="values", **kwargs):
        assert to in ["values", "keys", "both"]
        if to == "values":
            return self.__class__(*list((k, getattr(v, method)(*args, **kwargs)) for k, v in self.items()), parse=False)
        elif to == "keys":
            return self.__class__(*list((getattr(k, method)(*args, **kwargs), v) for k, v in self.items()), parse=False)
        elif to == "both":
            return self.__class__(*list(
                (getattr(k, method)(*args, **kwargs), getattr(v, method)(*args, **kwargs)) for k, v in self.items()),
                                  parse=False)

    def apply(self, func, *args, to="values", **kwargs):
        assert to in ["values", "keys", "both"]
        if to == "values":
            return self.__class__(*list((k, func(v, *args, **kwargs)) for k, v in self.items()), parse=False)
        elif to == "keys":
            return self.__class__(*list((func(k, *args, **kwargs), v) for k, v in self.items()), parse=False)
        elif to == "both":
            return self.__class__(*list(func((k, v), *args, **kwargs) for k, v in self.items()), parse=False)

    def inspect(self):
        return "|".join(f"{v} " + "{" + k + "}, " for k, v in self.items())

    def upper(self):
        return self.method_apply("upper")

    def lower(self):
        return self.method_apply("lower")

    @classmethod
    def from_pandoc_json(cls, ast_json: str) -> 'Text':
        """
        Creates a Text object from a Pandoc AST JSON string.
        """
        ast = json.loads(ast_json)
        content = cls._parse_elements(ast['blocks'])
        return cls(content)

    @classmethod
    def _parse_elements(cls, elements: List[Any]) -> List[Tuple[str, Any]]:
        """
        Recursively parses Pandoc AST elements into a list of tuples.
        """
        result = []
        for element in elements:
            type_ = element['t']
            if type_ == 'Header':
                level, _, inlines = element['c']
                text_content = cls._join_key_value(cls._parse_elements(inlines))
                result.append((f'header_h{level}', text_content))
            elif type_ == 'Para':
                text_content = cls._join_key_value(cls._parse_elements(element['c']))
                result.append(('p', text_content))
            elif type_ == 'Str':
                result.append(('text', element['c']))
            elif type_ == 'Space':
                result.append(('space', ' '))
            elif type_ == 'Emph':
                emph_text = cls._join_key_value(cls._parse_elements(element['c']))
                result.append(('emph', emph_text))
            elif type_ == 'Strong':
                strong_text = cls._join_key_value(cls._parse_elements(element['c']))
                result.append(('strong', strong_text))
            # ... Add more cases for other types of elements like lists, links, etc.
        return result

    @classmethod
    def _join_key_value(cls, elements: List[Tuple[str, Any]]) -> str:
        """
        Concatenates parsed text elements, handling spaces and formatting.
        """
        text = ''
        for element in elements:
            if element[0] in ['text', 'space']:
                text += element[1]
            else:
                # Add formatting tags or handle other types of elements
                text += f"<{element[0]}>{element[1]}</{element[0]}>"
        return text

    @classmethod
    def dict_to_ast(cls, dicts):
        import ast
        return ast.literal_eval(str(dicts).replace(":", ": ").replace("{", "{ ").replace("}", " }"))

    @classmethod
    def guess_format(cls, text):
        # Patterns for markup languages
        patterns = {
            'html': r'<!DOCTYPE html>|<html>',
            'markdown': r'^# .+|^- |\*\*[^*]+\*\*|__[^\_]+__',
            'latex': r'\\documentclass',
            # Add more patterns for other markup languages
        }

        # Check for markup language patterns
        for format_name, pattern in patterns.items():
            if re.search(pattern, text, re.MULTILINE):
                return format_name

        # Check for custom language
        if Text.detect_custom_language(text):
            return 'custom_language'

        # Default to plain texts if no patterns match
        return 'plain'

    @classmethod
    def detect_custom_language(cls, text):
        named_string_pattern = r'(\w+\{\:\w+\})|(`\w+`\{\:\w+\})|(\{\w+\:\w+\})|(\{\`\w+\`\:\w+\})|(\w+\{\`\`\:\})|(\{\:\w+\})'
        unnamed_string_pattern = r'(\{\w+\[\:\]\})|(\{\`\w+\`\:\})|(\{\`\w+\`\})|(\{\`\w+\`\:\})|\{\}|\`\`'
        full_pattern = fr'({named_string_pattern}|{unnamed_string_pattern})'
        # Match the pattern exactly; ^ and $ are the start and end of the string anchors respectively
        return bool(re.fullmatch(full_pattern, text))

    @classmethod
    def guess_language(cls, text):
        return Text.guess_format(text)

    @classmethod
    def load(cls, path: str, language: str = "str"):
        # Determine the caller's file path
        caller_frame = inspect.currentframe().f_back
        caller_file = inspect.getframeinfo(caller_frame).filename
        caller_dir = os.path.dirname(caller_file)

        # Resolve the relative path to an absolute path
        absolute_path = os.path.abspath(os.path.join(caller_dir, path))

        # Extract the file extension as the input format
        if language == "str":
            input_format = os.path.splitext(absolute_path)[1][1:]
        else:
            input_format = language

        # Read the file content
        with open(absolute_path, 'r', encoding="utf-8") as file:
            input_string = file.read()

        return cls(input_string, language=input_format, parse=input_format)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(str(self))

    from_file = load
    to_file = save
