from typing import List, Dict
from .backward_prompts import (
    BACKWARD_SYSTEM_PROMPT,
    CONVERSATION_TEMPLATE,
    CONVERSATION_START_INSTRUCTION_CHAIN,
    OBJECTIVE_INSTRUCTION_CHAIN,
    CONVERSATION_START_INSTRUCTION_BASE,
    OBJECTIVE_INSTRUCTION_BASE,
    EVALUATE_VARIABLE_INSTRUCTION
)
from langtorch import Activation
import logging

logger = logging.getLogger(__name__)

class BackwardActivation(Activation):
    @staticmethod
    def _construct_prompt(instruction_chain, backward_info):
        conversation = CONVERSATION_TEMPLATE.format(**backward_info)
        prompt = instruction_chain.format(conversation=conversation, **backward_info)
        prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        return prompt

    @staticmethod
    def _activation_backward(variables: List[Variable],
                             response: Variable,
                             prompt: str,
                             system_prompt: str,
                             is_chain: bool = True):
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "prompt": prompt,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value()
            }

            if is_chain:
                backward_info["response_gradient"] = response.get_gradient_text()
                instruction = CONVERSATION_START_INSTRUCTION_CHAIN
            else:
                instruction = CONVERSATION_START_INSTRUCTION_BASE

            backward_prompt = BackwardActivation._construct_prompt(instruction, backward_info)

            logger.info("Backward prompt", extra={"backward_prompt": backward_prompt})
            gradient_value = backward_engine(backward_prompt, system_prompt=BACKWARD_SYSTEM_PROMPT)
            logger.info("Gradient value", extra={"gradient_value": gradient_value})

            var_gradients = Variable(value=gradient_value,
                                     role_description=f"feedback to {variable.get_role_description()}")
            variable.gradients.add(var_gradients)
            variable.gradients_context[var_gradients] = {
                "context": CONVERSATION_TEMPLATE.format(**backward_info),
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description()
            }

            if response._reduce_meta:
                var_gradients._reduce_meta.extend(response._reduce_meta)
                variable._reduce_meta.extend(response._reduce_meta)

    @classmethod
    def backward_through_llm_chain(cls, variables, response, prompt, system_prompt, backward_engine):
        cls._activation_backward(variables, response, prompt, system_prompt, backward_engine, is_chain=True)

    @classmethod
    def backward_through_llm_base(cls, variables, response, prompt, system_prompt, backward_engine):
        cls._activation_backward(variables, response, prompt, system_prompt, backward_engine, is_chain=False)