from ..tt import TextModule


class CoTModule(TextModule):
    """Based on: zero-shot CoT (Kojima et al., 2022)"""

    def __init__(self, activation=None, key=None):
        super().__init__("{*}\nLet's think step by step.{:CoT}", activation, key)


CoT = CoTModule()


class CoT2Module(TextModule):
    """Based on: Large Language Models as Optimizers (Yang et al., 2023)"""

    def __init__(self, activation=None, key=None):
        super().__init__("{*}\nTake a deep breath and work on this problem step-by-step.{:CoT2}", activation, key)


CoT2 = CoT2Module()
