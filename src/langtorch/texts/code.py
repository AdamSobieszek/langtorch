import ast

from .text import Text


class Code(Text):
    @classmethod
    def str_formatter(cls, instance):
        return "\n".join([str(v) for v in instance.values()])

    @classmethod
    def _input_formatter(cls, instance):
        try:
            ast.parse(str(instance))
        except SyntaxError:
            raise ValueError(f"Invalid Python code: {instance}")
