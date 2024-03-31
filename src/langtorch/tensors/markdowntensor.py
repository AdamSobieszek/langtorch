from .texttensor import TextTensor
from ..texts import Markdown


class MarkdownTensor(TextTensor):
    ttype = Markdown  # replace parser

    def __new__(cls, *args, parse=True, **kwargs):
        if len(args) == 1 and isinstance(args[0], str) and parse:
            # If a single texts entry is given, split it using the items method of the Markdown parser.
            args = cls.ttype(args[0]).items()

        # Creating an instance as per the base class
        instance = super().__new__(cls, args, parse=True, **kwargs)

        # Reshape if the tensors shape's length is less than or equal to 2
        if len(instance.shape) <= 2 and (len(instance.shape) == 0 or instance.shape[-1] != 1):
            instance = instance.view(-1, 1)

        return instance

    def headers_to_keys(self):
        """
        Removes headers entries of the MarkdownTensor and places them as keys of their following paragraphs.
        If multiple header levels are present, the keys are stacked h1.h2.h3...

        Returns
        -------
        MarkdownTensor
                """
        return self.__new__(self)  # TODO
