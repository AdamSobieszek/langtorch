from .text import Text


class Thread(Text):
    allowed_keys = ["user", "assistant"]

    @classmethod
    def str_formatter(cls, instance):
        return "\n".join([f"{k}: {cls._concatenate_terminals(v)}" for k, v in instance])
