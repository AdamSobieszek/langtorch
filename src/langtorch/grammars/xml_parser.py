import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import Union, Tuple, List


def xml_to_ast(xml_string: str) -> Union[Tuple[str, Union[str, Tuple]], Tuple]:
    """ Converts an XML string to a structured tuple representation of its AST. """
    # Parse the XML string into an ElementTree
    try:
        root = ET.fromstring(xml_string)
    except ParseError:
        root = ET.fromstring("<xml_temp_root>" + xml_string + "</xml_temp_root>")

    def recurse(element):
        """ Recursively traverse the XML elements and build a tuple structure. """
        result = []
        # Process any leading text of the element that is just text
        if element.text and len(list(element)) == 0:
            result.append(element.text)
        elif element.text:
            result.append(("", element.text))
        # Iterate over all children elements
        for child in element:
            result.append((child.tag, recurse(child)))
            # Process any text between this child and the next child (tail text)
            if child.tail:
                result.append(("", child.tail))
        return result if len(result) > 1 else result[0] if result else ""

    # Handle the case where the root element may have text or multiple elements
    parsed_result = recurse(root)
    # If the root itself contains only one element and no leading text, return it directly
    if root.tag != "xml_temp_root":
        return ((root.tag, parsed_result),)
    elif isinstance(parsed_result, str):
        return (parsed_result,)
    elif isinstance(parsed_result, tuple) and len(parsed_result) == 1:
        return (parsed_result,)
    else:
        return tuple(parsed_result)

# Test cases:
# xml_string1 = "<keyA>text text1</keyA><keyA>text text1</keyA>"
# xml_string2 = "<keyA>smth<keyA>text text1</keyA></keyA>"
# xml_string3 = "<root>" + xml_string1 + "text with an empty key" + xml_string2 + "</root>"
#
# print(xml_to_ast(xml_string1))
# print(xml_to_ast(xml_string2))
# print(xml_to_ast(xml_string3))

# Outputs:
# (('keyA', 'text text1'), ('keyA', 'text text1'))
# (('keyA', [('', 'smth'), ('keyA', 'text text1')]),)
# (('root', [('keyA', 'text text1'), ('keyA', 'text text1'), ('', 'text with an empty key'), ('keyA', [('', 'smth'), ('keyA', 'text text1')])]),)


# def ast_to_xml(ast: Union[Tuple[str, Union[str, Tuple]], Tuple, List]) -> str:
#     """Converts a structured tuple representation of an XML AST back to an XML string."""
#
#     def recurse(node: Union[str, Tuple, List]) -> str:
#         """Recursively traverse the AST and build the XML string."""
#         if isinstance(node, str):
#             return node
#         elif isinstance(node, tuple):
#             tag, content = node
#             if tag == '':  # Handle empty tags (text nodes)
#                 return content
#             if isinstance(content, str):
#                 return f"<{tag}>{content}</{tag}>"
#             elif isinstance(content, (tuple, list)):
#                 inner_content = ''.join(recurse(child) for child in content)
#                 return f"<{tag}>{inner_content}</{tag}>"
#         elif isinstance(node, list):
#             return ''.join(recurse(child) for child in node)
#         else:
#             raise ValueError(f"Unexpected node type: {type(node)}")
#
#     if isinstance(ast, tuple) and len(ast) == 1:
#         ast = ast[0]
#
#     # Handle multiple top-level elements
#     if isinstance(ast, (tuple, list)) and all(isinstance(item, tuple) for item in ast):
#         return ''.join(recurse(item) for item in ast)
#
#     return recurse(ast)


# Test cases:
# ast1 = (('keyA', 'text text1'), ('keyA', 'text text1'))
# ast2 = (('keyA', [('', 'smth'), ('keyA', 'text text1')]),)
# ast3 = (('root', [('keyA', 'text text1'), ('keyA', 'text text1'), ('', 'text with an empty key'),
#                   ('keyA', [('', 'smth'), ('keyA', 'text text1')])]),)
#
# print(ast_to_xml(ast1))
# print(ast_to_xml(ast2))
# print(ast_to_xml(ast3))
