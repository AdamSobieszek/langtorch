import torch
from torch.optim import Optimizer

from langtorch import TextTensor


class TextOptimizer(Optimizer):
    def __init__(self, params, defaults):
        super(TextOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if not isinstance(param, TextTensor):
                    raise ValueError("Optimizer parameters must be of type TextTensor")
                if param.grad is None:
                    continue
                grad = param.grad
                # Perform the optimization step to update 'param.data'
                # This is where the custom optimization logic would go
                # For demonstration, let's just subtract a fraction of the gradient
                param.data.add_(-group['lr'], grad)

        return loss

    def add_param_group(self, param_group):
        """Add a parameter group to the Optimizer's param_groups.

        This would be overridden only if we need to enforce specific types within param groups.
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")

        params = param_group['params']
        if isinstance(params, TextTensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError("parameter group cannot be a set")
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, TextTensor):
                raise ValueError("Optimizer parameters must be of type TextTensor")
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf TextTensor")

        self.param_groups.append(param_group)
