from pyparsing import ParseResults

from .langtorch_default_parser import LangTorchGrammarParser
from .pandoc import pandoc_to_ast


def pandoc_parser(txt, language="md"):
    if language == "str":
        return (txt,)
    try:
        return pandoc_to_ast(txt, language)
    except Exception as e:
        print(f"Error parsing with pandoc: {e}")
        return (txt,)


def langtorch_parser(txt):
    parsed_result = LangTorchGrammarParser.parseString(txt)
    arg = [(res.key if "key" in res else "", res.value if "value" in res else "") if isinstance(res,
                                                                                                ParseResults) else res
           for res in parsed_result]
    return arg


class Parsers:
    def __getitem__(self, item):
        return {
            "langtorch-f-string": langtorch_parser,
            "f-string": langtorch_parser,
            True: langtorch_parser,
            False: lambda txt: txt
        }.get(item, lambda txt: pandoc_parser(txt, item))


language_to_parser = Parsers()
