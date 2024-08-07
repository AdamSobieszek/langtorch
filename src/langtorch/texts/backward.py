from .text import Text


class BackwardText(Text):
    language = 'backward'

    def __str__(self):
        return self.__class__.str_formatter(self, 'xml', True)