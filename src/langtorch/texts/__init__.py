from .chat import Chat, ChatML, HumanMessage, AIMessage, User, Assistant
from .code import Code
from .markdown import Markdown
from .text import Text
from .string import String
from .xml import XML
from .backward import BackwardText

def text_factory(language: str):
    def new(cls, *substrings, **kwargs):
        if 'parse' not in kwargs:
            kwargs['parse'] = language
        if 'language' not in kwargs:
            kwargs['language'] = language
        return super(cls, cls).__new__(cls, *substrings, **kwargs)
    return type(language.capitalize(), (Text,), {'__new__':new, 'language': language})