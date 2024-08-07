from pyparsing import ParseResults
import logging

from .langtorch_default_parser import LangTorchGrammarParser, fix_double_brackets, fix_substitution
from .pandoc import pandoc_to_ast
from .xml_parser import xml_to_ast


def pandoc_parser(txt, language="md"):
    if language == "str":
        return (txt,)
    try:
        return pandoc_to_ast(txt, language)
    except Exception as e:
        print(f"Error parsing with pandoc: {e}")
        return (txt,)


def langtorch_parser(txt):
    try:
        txt_ = fix_double_brackets(txt)
        txt_ = fix_substitution(txt_)
    except:
        logging.debug(f"The double bracket containing string could not be parsed: {txt}")
        txt_ = txt
    parsed_result = LangTorchGrammarParser.parseString(txt_)
    arg = [(res.key if "key" in res else "", res.value if "value" in res else "") if isinstance(res,
                                                                                                ParseResults) else res
           for res in parsed_result]
    return arg


def xml_parser(txt):
    try:
        return xml_to_ast(txt)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return (txt,)


class Parsers:
    def __getitem__(self, item):
        return {
            "langtorch-f-string": langtorch_parser,
            "f-string": langtorch_parser,
            "xml": xml_parser,
            True: langtorch_parser,
            False: lambda txt: txt
        }.get(item, lambda txt: pandoc_parser(txt, item))


language_to_parser = Parsers()
