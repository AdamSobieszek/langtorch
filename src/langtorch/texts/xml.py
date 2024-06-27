from typing import List, Union, Tuple, Any


from .text import Text

class XML(Text):
    language = 'xml'

    def __new__(cls, *substrings, parse="xml", language="xml", **kwargs):
        # Prepare content appropriately before calling the superclass's __new__
        instance = super().__new__(cls, *substrings, parse=parse, language=language, **kwargs)
        return instance