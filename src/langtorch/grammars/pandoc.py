import json
import os
from typing import Optional

import pypandoc


def join_str_types(pandoc_tuples):
    """
    Joins all 'Str' types to each other if the second item in the tuple is a string.

    Args:
        pandoc_tuples (list): The list of Pandoc tuples.

    Returns:
        list: The modified list of Pandoc tuples.
    """
    if isinstance(pandoc_tuples, str):
        return pandoc_tuples
    elif isinstance(pandoc_tuples, tuple):
        if pandoc_tuples[0] == 'Str':
            return pandoc_tuples[1]
    elif len(pandoc_tuples) == 1:
        return join_str_types(pandoc_tuples[0])
    elif all(isinstance(x, str) for x in pandoc_tuples):
        return "".join(pandoc_tuples)
    result = []
    temp_str = ""


    for item in pandoc_tuples:
        if not item:
            continue
        if isinstance(item, str):
            temp_str += item
        elif isinstance(item, tuple):
            if item[0] == 'Space':
                temp_str += " "
            elif item[0] == 'SoftBreak':
                temp_str += "\n"
            elif item[0] == 'DoubleQuote':
                temp_str += "\""
            elif item[0] == 'SingleQuote':
                temp_str += "''"
            elif item[0] == 'LineBreak':
                result.append(temp_str)
                temp_str = ""
            elif item[0] == 'Quoted':
                item = item[1]
                if isinstance(item, list):
                    if len(item) > 2:
                        if temp_str:
                            result.append(temp_str)
                            temp_str = ""
                        result.extend(join_str_types(item))
                        continue
                    else:
                        item = join_str_types(item)
                        if isinstance(item, str): # in case there are exceptions to this pandoc pattern
                            item += item[0]  # Add closing quote
                if isinstance(item, str):
                    temp_str += item
                else:
                    if temp_str:
                        result.append(temp_str)
                        temp_str = ""
                    result.append(item)
            elif item[0] in ['Str'] and isinstance(item[1], str): #, 'Emph', 'Bold'?
                temp_str += item[1]
            else:
                if temp_str:
                    result.append(temp_str)
                    temp_str = ""

                if item[0] in ['Plain']:
                    item = item[1]
                elif item[0] in ['OrderedList'] and len(item[1]) == 2 and isinstance(item[1][0], list):
                        item = (item[0], item[1][1])
                result.append(item)
        elif isinstance(item, list):
            item = join_str_types(item)
            if isinstance(item, str):
                temp_str += item
            else:
                result.append(temp_str)
                temp_str = ""
                result.append(item)
        else:
            if temp_str:
                result.append(temp_str)
                temp_str = ""

            result.append(item)
    if temp_str:
        result.append(temp_str)
    return result


def pandoc_dict_to_tuple(d):
    if isinstance(d, list):
        return join_str_types([pandoc_dict_to_tuple(x) for x in d])
    if isinstance(d, dict):
        if "c" not in d:
            return (d["t"], "")
        if isinstance(d["c"], str):
            return (d["t"], d["c"])
        else:
            # Identify Header size
            if len(d["c"]) == 3 and isinstance(d["c"][0], int):
                d["t"] += str(d["c"][0])
            return (d["t"], pandoc_dict_to_tuple(d["c"]))


def pandoc_to_ast(input_string: str, input_format: str) -> str:
    """
    Converts a given string into a Pandoc AST JSON format.

    Args:
        input_string (str): The input string to be converted.
        input_format (str): The format/language of the input string.

    Returns:
        str: A string containing Pandoc AST in JSON format.
    """

    # Convert the input string to Pandoc JSON format
    output = json.loads(pypandoc.convert_text(input_string, 'json', format=input_format))
    return pandoc_dict_to_tuple(output["blocks"])


def pandoc_to_ast_from_file(file_path: str, language: Optional[str] = None) -> str:
    """
    Converts a given file into a Pandoc AST JSON format.

    Args:
        file_path (str): The path to the file to be converted.

    Returns:
        str: A string containing Pandoc AST in JSON format.
    """

    # Extract the file extension as the input format
    if language is None:
        input_format = os.path.splitext(file_path)[1][1:]
    else:
        input_format = language

    # Read the file content
    with open(file_path, 'r') as file:
        input_string = file.read()

    # Convert the input string to Pandoc JSON format
    output = json.loads(pypandoc.convert_text(input_string, 'json', format=input_format))
    return pandoc_dict_to_tuple(output["blocks"]), input_format
