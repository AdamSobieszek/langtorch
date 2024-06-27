from .texttensor import TextTensor
from ..texts import XML


class LossTensor(TextTensor):
    ttype = XML

    def __new__(cls, *args, parse=True, **kwargs):
        instance = super().__new__(cls, *args, parse=True, **kwargs)

        return instance
