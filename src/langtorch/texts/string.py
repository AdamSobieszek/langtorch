from .text import Text
from ..grammars import text_content_ast as ast


class String(Text):
    language = 'str'
    allowed_keys = [""]

    @classmethod
    def constructors(*args, parse=False):
        return [("", "".join(args))]

    @classmethod
    def _to_ast(cls, content, parser=False, is_tuple=True):
        content = ast.to_ast(content, parser, is_tuple)
        return content if not is_tuple else (super().str_formatter(content),)
