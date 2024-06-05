import json
import logging
from typing import Optional, Union, Tuple, List

import numpy as np
import torch

import langtorch.utils
from langtorch.api.call import chat
from langtorch.decorators import set_defaults_from_ctx
from langtorch.tensors import TextTensor
from .textmodule import TextModule


class ActivationOpenAI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the forward pass computation
        ctx.save_for_backward(input)
        # TODO differntiable chat

    @staticmethod
    def backward(ctx, grad_output):
        # Perform the backward pass computation
        input, = ctx.saved_tensors
        return input


class Activation(TextModule):
    input_class = TextTensor
    output_class = TextTensor

    @set_defaults_from_ctx
    def __new__(cls, model: str = "default",
                system_message: str = None,
                provider: str = "openai",
                cache: bool = False,
                keep_history: bool = False,
                T: float = 0.8,
                tools: Optional[List[dict]] = None,
                key: Optional[str] = None,
                backward_prompt: Union[str, TextTensor] = "default",
                *args, **kwargs):
        if any([arg == "default" for arg in [model, system_message]]):
            print(f"model = {model}, system_message = {system_message}")
            raise ValueError(
                "Activation requires a model and system_message,, but these were not loaded from defaults or passed.")
        if cls is Activation:
            # Return an instance of the OpenAI subclass instead of Activation
            # In the future more subclasses of Activation will be added here
                activation = OpenAI(model=model,
                              system_message=system_message,
                              cache=cache,
                              keep_history=keep_history,
                              T=T,
                              tools=tools,
                              key=key,
                              backward_prompt=backward_prompt,
                              *args, **kwargs)
                activation.provider = provider
                return activation
        else:
            # If a subclass of Activation is being instantiated, proceed as normal
            return super().__new__(cls)


class OpenAI(TextModule):
    input_class = TextTensor
    output_class = TextTensor
    provider = "openai"

    def __init__(self,
                 model: Union[str, TextTensor] = "gpt-3.5-turbo",
                 system_message: Union[
                     str, TextTensor] = None,
                 backward_prompt: Union[str, TextTensor] = "",
                 cache: bool = False,
                 keep_history: bool = False,
                 T: float = 0.8,
                 tools: Optional[List[dict]] = None,
                 parse_output=False,
                 *args, **kwargs):
        super(OpenAI, self).__init__()
        self.system_message = str(system_message) if system_message is not None else None
        self.model = model
        self.backward_prompt = backward_prompt
        self.keep_history = True if keep_history or kwargs.get('echo', False) else False
        if not 'temperature' in kwargs:
            kwargs['temperature'] = T
        cache = cache | (kwargs['temperature'] == .0)
        self.cache = cache
        self.kwargs = kwargs

        self.tool_jsons = tools if isinstance(tools, list) else [tools] if tools is not None else None

        if self.tool_jsons is not None:
            self.tool_jsons = [json.dumps(tool, ensure_ascii=False) if isinstance(tool,
                                                                                  dict) else langtorch.api.tools.generate_schema_from_function(
                tool) if callable(tool) else str(tool) for tool in self.tool_jsons]

        self.n = kwargs.get('n', 1)
        self.parse_output = parse_output

    def forward(self, input: TextTensor):
        if isinstance(input, list):
            shape = (-1, self.n) if self.n > 1 else -1
            input = TextTensor(input)
        elif isinstance(input, TextTensor):
            shape = tuple([self.n]+[m for m in input.shape]) if self.n > 1 else tuple(input.shape)
        else:
            raise TypeError("Activation handles only lists and TextTensors")

        input = list(input.content.flat)
        system_messages = [self.system_message] * len(input)

        # Transform chat input into (role, content) pairs
        for i, m in enumerate(input):
            if "system" in m.keys():
                system_messages[i] = m.loc["system"].values()[-1]
                m = m.loc[["user", "assistant"]]
            if np.all([key in ["user", "assistant"] for key in m.keys()]):
                input[i] = m.items()
            elif "user" not in m.keys() or "assistant" not in m.keys():
                # NOT STRICT
                input[i] = [("user", str(m))]
                logging.debug(
                    f"A text without exclusively 'assistant' ot 'user' keys was passed to OpenAI Chat. Assuming the whole Text is one user message: = '{str(m)[:25]}'... ")
            else:
                raise ValueError(
                    f"Invalid input to OpenAI Chat. Ambiguous input: some but not all items in TextTensor entries have keys 'user' or 'assistant'. Change the input into a valid chat, or the input was a single user message remove these keys or use entry.add_key_('user'): \nentry.items()=={m.items()}")
        logging.debug(f"Chat api input: {input}")
        logging.debug(f"Chat api unique system messages: {set(system_messages)}")

        result = chat(input, system_messages, model=self.model, provider=self.provider, cache=self.cache, as_str=True, tools=self.tool_jsons,
                      **self.kwargs)

        assert result is not None
        if not self.keep_history:
            return TextTensor(result, parse=self.parse_output).reshape(shape)
        else:
            return input + TextTensor(result, parse=self.parse_output).reshape(shape)


ChatGPT, GPT = OpenAI, OpenAI

GPT4 = lambda *args, **kwargs: Activation("gpt-4", *args, **kwargs)


class TODO_LLM(torch.nn.Module):
    def __init__(self,
                 model: Union[str, TextTensor] = "gpt-3.5-turbo",
                 system_message: Union[
                     str, TextTensor] = "You are an expert, who answers only with the requested texts. Keep it short.",
                 backward_prompt: Union[str, TextTensor] = "",
                 cache: bool = False,
                 keep_history: bool = False,
                 T: float = 0.8,
                 tools: Optional[dict] = None,
                 *args, **kwargs):
        super(LLM, self).__init__()
        self.system_message = str(system_message)
        self.model = model
        self.backwad_prompt = backward_prompt
        if not 'temperature' in kwargs:
            kwargs['temperature'] = T
        cache = cache | (kwargs['temperature'] == .0)
        self.cache = cache
        self.tools = tools if isinstance(tools, list) else [tools] if tools is not None else None
        self.n = kwargs.get('n', 1)
        self.kwargs = kwargs

    def forward(self, input: TextTensor):
        raise NotImplementedError("LLM Requires a forward method")


class TODO_MultiheadAttention(torch.nn.Module):
    r"""

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        from torch.nn import Parameter
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
            self,
            query: TextTensor,
            key: TextTensor,
            value: TextTensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[TextTensor, Optional[torch.Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
            Default: ``False``.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
