import torch
from torch.optim import Optimizer
from typing import Any, Dict, Iterable, List, TypeAlias, Union
from langtorch import TextTensor, Activation


class TextOptimizer(Optimizer):
    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 defaults: Dict[str, Any] = {},
                 activation = None) -> None:
        """Base class for all text optimizers.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        activation: (langtorch.Activation): an activation LLM to use during an optimizer step.
    """
        self.activation = activation
        super(TextOptimizer, self).__init__(params, defaults)
        for group in self.param_groups:
            for param in group['params']:
                assert isinstance(param,TextTensor), f"All params have to be TextTensors, but there is param = {param}"


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
                    raise ValueError("TextOptimizer parameters must be of type TextTensor")
                if param.grad is None:  # We don't update the parameters that have no gradients
                    continue
                grad = param.grad
                # Perform the optimization step to update 'param.content'
                param.content = (param + "\ngrad:\n").content + grad.content

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


class TextualGradientDescent(TextOptimizer):
    def __init__(self,
                 params,
                 activation: Activation = None):
        """TextualGradientDescent optimizer"""
        super().__init__(params, {})

        if isinstance(activation, str):
            activation = Activation(activation)

    def _update_prompt(self, param):
        optimizer_information = {
            "variable_desc": param.get_role_description(),
            "variable_value": param.value,
            "variable_grad": param.get_gradient_and_context_text(),
            "variable_short": param.get_short_value()
        }

        prompt = construct_tgd_prompt()
        return prompt

    def step(self):
        """
        Perform a single optimization step.
        This method updates the parameters of the optimizer by generating new text using the engine and updating the parameter values accordingly.
        It also logs the optimizer response and the updated text.
        Returns:
            None
        """
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)
            parameter.set_value(
                new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip())
            # if self.verbose:
            #     print("-----------------------TextualGradientDescent------------------------")
            #     print(parameter.value)
