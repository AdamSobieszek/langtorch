from copy import deepcopy
from typing import Any, Optional, Union, List, Callable, Iterable

import numpy as np
import torch
from torch.types import _TensorOrTensors
from pyparsing import ParseException
import tiktoken
import os

import langtorch
from .. import utils
from ..api.call import get_embedding
from ..autograd import make_grads
from ..texts import Text
from ..grammars import formatters
from ..tt.functional import AddTextTensor, MulTextTensor, PermuteTextTensor, FormatTextTensor, JoinTextTensor, \
    ReshapeTextTensor, SplitTextTensor


# Metaclass to combine _TensorMeta and the instance check override for Parameter.
class _ParameterMeta(torch._C._TensorMeta):
    # Make `isinstance(t, Parameter)` return True for custom tensors instances that have the _is_param flag.
    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (
                isinstance(instance, torch.Tensor) and getattr(instance, '_is_param', False))


def chararray_to_TextArray(arr, ttype=Text, shape=None, **kwargs):
    if isinstance(arr, np.ndarray):
        try:
            if all([isinstance(a, ttype) for a in arr.flat]):
                return arr
        except:
            pass

    arr = np.array(arr, dtype=object)
    text_arr = [ttype(a, **kwargs) for a in arr.flat]
    text_arr = np.array(text_arr, dtype=object).reshape(arr.shape if shape is None else shape)
    return text_arr






class TextTensor(torch.Tensor, metaclass=_ParameterMeta):
    """
    TextTensor is a specialized subclass of torch.Tensor designed for handling and manipulating textual data within the LangTorch framework. Each entry in a TextTensor is a structured Text object, enabling complex text operations and transformations, including natural language processing tasks and interaction with large language models (LLMs).

    TextTensor supports standard tensor operations adapted for text, such as concatenation and reshaping, alongside unique text-specific operations like prompt formatting through multiplication. It seamlessly integrates with PyTorch, allowing developers to leverage familiar tensor operations while working with textual data.

    Attributes:
        ttype (Class): Specifies the text type for entries in the TextTensor, ttype should be a subclass of the Text class.
        _embedding_model (Union[str, TextModule], optional): Name of an OpenAI embedding model or TextModule to convert TextTensor entries to embeddings.
        _tokenizer (Tokenizer, optional): Tokenizer for converting TextTensor entries to tokens suitable for LLMs.
        parse (bool, optional): Controls automatic parsing of text entries. Default is 'auto', which decides based on context.

    Special Operations:
        - Addition (`+`): Concatenates text entries.
        - Multiplication (`*`): Formats prompt templates with values from another TextTensor or dictionary.

    Keyword Arguments:
        content (str, Text, list, dict): Initial content for the TextTensor. Can be a single string, a Text object, a list of strings or Text objects, or a dictionary for named entries.
        metadata (dict, optional): Additional metadata for the TextTensor.
         requires_grad (bool, optional): Indicates whether the tensor should track gradients.
        is_gradient (bool, optional): Marks the tensor as a gradient tensor.
        is_param (bool, optional): Marks the tensor as a parameter tensor.
        **kwargs: Additional keyword arguments passed to the underlying torch.Tensor.

    Examples:
        Creating a TextTensor with prompt templates:
        >>> tt = TextTensor(["Hello, {name}!"])
        >>> tt * TextTensor({"name": "World"})
        TextTensor(["Hello, World!"])

        Concatenating TextTensors:
        >>> tt1 = TextTensor(["Hello"])
        >>> tt2 = TextTensor([", World!"])
        >>> tt1 + tt2
        TextTensor(["Hello, World!"])

    Note:
        - Ensure proper key matching when performing operations that rely on named entries.
        - Consider setting `requires_grad` appropriately for training LLMs or gradient-based operations.

    Warning:
        - Incorrect usage of keys or mismatched shapes during operations may lead to unexpected results.
    """

    _ttype = Text  # The class of a Tensor entry, replaced in subclasses. May in the future move this logic to the metaclass
    _embedding_model = "text-embedding-3-small"  # TextTensor to Tensor with an embedding model
    _tokenizer = 'cl100k_base'  # TextTensor to Tensor with a tokenizer (tiktoken or transformers)
    _embedding, _tokens = None, None
    _tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
    _tiktoken_tokenizers = ['cl100k_base', 'p50k_base', 'r50k_base']
    parse = 'auto'
    is_gradient = False
    is_param = False

    def __new__(cls, content="", parse=True, metadata=None,
                ttype: Text = None,
                embedding_model: Union[str, Callable] = None,
                tokenizer: Union[str, Callable] = None,
                tokenizer_kwargs: dict = None,
                requires_grad: bool = False,
                is_gradient: bool = False,
                is_param: bool = False, **kwargs):
        embedding = None
        if metadata is None:
            metadata = dict()
        for attr in ["content", "embedding"]:
            if not attr in metadata:
                metadata[attr] = eval(attr)


        if isinstance(metadata["content"], dict):
            metadata["content"] = cls._dict_to_tt(metadata["content"], parse=parse)

        def replace_tuple_with_ttype(tpl):
            nonlocal cls
            return cls._ttype(*tpl)
        # Replace tuples with Text entries
        metadata["content"] = cls._recursive_walk_replace(metadata["content"], tuple, replace_tuple_with_ttype)
        # Set content to be an object array with cls.text_class entries
        metadata["content"] = cls.content_to_object_array(metadata["content"], parse=parse)
        # Apply input formatter
        metadata["content"] = cls.input_formatter(metadata["content"])

        tensor = super().__new__(cls, torch.arange(metadata["content"].size, dtype=torch.float32).reshape(
            metadata["content"].shape), **kwargs)

        tensor.metadata = metadata
        tensor._is_param = is_param
        tensor.requires_grad = requires_grad or is_param
        tensor._is_gradient = is_gradient
        assert tensor._content is not None

        # Apply linter
        tensor = cls.linter(tensor)
        if ttype is not None:
            tensor.ttype = ttype
        if embedding_model is not None:
            tensor.embedding_model = embedding_model
        if tokenizer is not None:
            tensor.tokenizer = tokenizer
        if tokenizer_kwargs is not None:
            tensor._tokenizer_kwargs = tokenizer_kwargs

        return tensor

    @classmethod
    def input_formatter(cls, content):
        # Default implementation, to be overridden by subclasses if needed
        return content

    @classmethod
    def linter(cls, tensor):
        # Default implementation, ensures correct class of Tensor content entries
        tensor.content = chararray_to_TextArray(tensor.content, cls._ttype)
        return tensor

    @classmethod
    def _recursive_walk_replace(cls, obj: Any, target_class: type, func: Callable[[Any], Any]) -> Any:
        """
        Recursively walks through any iterable structure, finds all instances of target_class,
        and replaces them with the result of func applied to them.

        :param obj: The object to walk through.
        :param target_class: The class to look for in the structure.
        :param func: The function to apply to instances of target_class.
        :return: A new object with the replacements made.
        """
        if isinstance(obj, target_class):
            return func(obj)
        elif isinstance(obj, dict):
            return {k: cls._recursive_walk_replace(v, target_class, func) for k, v in obj.items()}
        elif isinstance(obj, (list, set)):
            cls2 = type(obj)
            return cls2(cls._recursive_walk_replace(item, target_class, func) for item in obj)
        elif isinstance(obj, tuple):
            return func(obj)  # Directly replace tuple with _ttype
        else:
            return obj

    @classmethod
    def _dict_to_tt(cls, d, parse):
        if not d:
            return None
        def item_to_tt(cls, k, v, parse):
            tt = cls.__new__(cls, v, parse=parse).add_key(k)
            return tt

        items = iter(d.items())
        first_k, first_v = next(items)
        content = item_to_tt(cls, first_k, first_v, parse=parse)
        first_shape = content.shape
        try:
            for k, v in items:
                content = content + item_to_tt(cls, k, v, parse=parse)
        except Exception as e:
            raise ValueError(
                f"Could not convert dictionary of TextTensors to a single. Failed to add with broadcasting entry:\n{k}: {v}\nException: {e}")
        return content
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return id(self)

    @classmethod
    def content_to_object_array(cls, input, **kwargs):
        try:
            return input.content if isinstance(input, TextTensor) \
                else np.array(input, dtype=object) if isinstance(input, Text) \
                else chararray_to_TextArray(input, cls._ttype, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Could not convert input to TextTensor content. Failed to convert input to an array:\n{e}")

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, content):
        self._content = TextTensor.content_to_object_array(content)
        assert self._content is not None
    @property
    def embedding_model(self):
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, embedding_model):
        self._embedding_model = embedding_model
        assert self._content is not None

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: Union[torch.Tensor, List[float], np.ndarray, tuple]):
        if embedding is None:
            self._embedding = None
            return
        assert isinstance(embedding, torch.Tensor) or isinstance(embedding, np.ndarray) or isinstance(embedding,
                                                                                                      list) or isinstance(
            embedding, list), "Value of embedding should be a Tensor, array or list"

        self._embedding = embedding if isinstance(embedding, torch.Tensor) else torch.vstack(
            [m if isinstance(m, torch.Tensor) else torch.tensor(m) for m in embedding])
        try:
            self._embedding = self._embedding.reshape(*self.content.shape, -1)
        except ValueError:
            raise ValueError("The shape of the embedding does not match the size of the TextTensor")
        self._metadata["embedding"] = self._embedding

    @property
    def embeddings(self):
        raise AttributeError("The attr with embeddings is called 'embedding' not 'embeddings'")

    @property
    def tokens(self):
        if not hasattr(self, "_tokens"):
            if self.tokenizer is not None:
                self._tokens = self.tokenizer(self)
            else:
                raise AttributeError("Tokens unavailable as no tokenizer has been set")
        return self._tokens

    input_ids = tokens

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        if not isinstance(tokenizer, str) and not callable(tokenizer):
            raise ValueError("Tokenizer must be a string with a tokenizer name or a callable tokenizer")
        self._tokenizer = tokenizer

    @property
    def tokenizer_kwargs(self):
        return self._tokenizer_kwargs

    @tokenizer_kwargs.setter
    def tokenizer_kwargs(self, tokenizer_kwargs):
        self._tokenizer_kwargs = dict(tokenizer_kwargs)

    def tokenize(self, tokenizer=None, **kwargs):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        assert self.tokenizer is not None, "No tokenizer set"
        kwargs = {**self._tokenizer_kwargs, **kwargs} # Kwargs only for transformers AutoTokenizer

        if isinstance(self.tokenizer, str):
            if self.tokenizer in self._tiktoken_tokenizers:
                # Load the specific encoding from tiktoken
                tokenize_function = lambda content: utils.pad_sequences_and_create_masks([tiktoken.get_encoding(self.tokenizer).encode(x) for x in content])
            else:
                # Use transformers' AutoTokenizer for other models
                from transformers import AutoTokenizer
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                tokenize_function = lambda content: AutoTokenizer.from_pretrained(self.tokenizer)(content, **kwargs)
        else:
            tokenize_function = self.tokenizer

        content = [str(m) for m in self.content.flat]
        self.tokens = tokenize_function(content)

        return self.tokens

    @property
    def ttype(self):
        return self._ttype

    @ttype.setter
    def ttype(self, text_class):
        if issubclass(text_class, Text):
            self._ttype = text_class
        else:
            if callable(text_class):
                raise ValueError("ttype must be a subclass of Text or its name as a string")
            text_class = str(text_class)
            assert text_class in dir(langtorch.texts), f"Text class {text_class} not found in langtorch.texts"

            self._ttype = getattr(langtorch.texts, text_class)
        self.content = np.array([self._ttype(t) for t in self.content.flat], dtype=object).reshape(self.content.shape)
        # iter over multidimensional self._content
        for index, text_entry in np.ndenumerate(self._content):
            self._content[index].language = self._ttype.language

    functions_on_texts = [torch.cat, torch.concat, torch.concatenate, torch.vstack, torch.stack, torch.hstack,
                          torch.squeeze, torch.unsqueeze, torch.reshape, torch.swapaxes, torch.sum, torch.nonzero]
    functions_on_embeddings = [torch.mean, torch.cosine_similarity]
    functions_on_tokens = []

    @classmethod
    def __torch_function__(cls, func: Callable, types: Iterable, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Handle functions that need custom behavior
        if func in cls.functions_on_texts:
            return cls._handle_functions_on_texts(func, *args, **kwargs)
        if func in cls.functions_on_embeddings:
            return cls._handle_functions_on_embeddings(func, *args, **kwargs)
        if func in cls.functions_on_tokens:
            return cls._handle_functions_on_tokens(func, *args, **kwargs)

        # Default behavior for unhandled functions
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def _handle_functions_on_texts(self, func: Callable, *args, **kwargs):
        if func.__name__ in dir(langtorch):
            return getattr(langtorch, func.__name__)(*args, **kwargs)
        # Fallback if the function is not recognized
        raise NotImplementedError("This torch function has no langtorch equivalent yet")

    @classmethod
    def _handle_functions_on_embeddings(self, func: Callable, *args, **kwargs):
        """Substitutes the TextTensor entries with their embeddings and applies the function."""

        def apply_to_texttensors(func, args, kwargs):
            args2, kwargs2 = [], {}
            first = None
            for i, arg in enumerate(args):
                if isinstance(arg, TextTensor):
                    args2.append(func(arg))
                    if first is None:
                        first = arg
                else:
                    args.append(arg)
            for k, arg in kwargs.items():
                if isinstance(arg, TextTensor):
                    kwargs2[k] = func(arg)
                    if first is None:
                        first = arg
                else:
                    kwargs2[k] = arg

            return first, args2, kwargs2

        def apply_embed(tensor, model = None) -> torch.Tensor:
            tensor.embed(model=model)
            assert isinstance(tensor.embedding, torch.Tensor)
            return tensor.embedding

        first, args, kwargs = apply_to_texttensors(lambda t:t, args, kwargs)
        model = kwargs.pop("model", None)
        if model is None:
            model = first.embedding_model
        first, args, kwargs = apply_to_texttensors(lambda t:apply_embed(t, model), args, kwargs)

        if func is torch.cosine_similarity:
            kwargs["dim"] = -1
            # args = [a.reshape(-1) for a in args]
            return torch.cosine_similarity(*args, **kwargs)

        return func(*args, **kwargs)

    @classmethod
    def _handle_functions_on_tokens(self, func: Callable, *args, **kwargs):
        # Fallback if the function is not recognized
        raise NotImplementedError("This torch function has no langtorch equivalent yet")

    @property
    def loc(self):
        """
        Provides key-based indexing for TextTensor, leveraging the `loc` method of the Text entries.
        You can access and manipulate sub-elements of the TextTensor based on keys.
        """

        class LocIndexer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __getitem__(self, key):
                result = [text.loc[key] for text in self.tensor.content.flat]
                shape = self.tensor.content.shape
                return TextTensor(np.array(result, dtype=object).reshape(shape), parse=False)

            def __setitem__(self, key, value):
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=object).reshape(self.tensor.content.shape)
                for index, text_entry in np.ndenumerate(self.tensor.content):
                    text_entry.loc[key] = value[index]

        return LocIndexer(self)

    @property
    def iloc(self):
        """
        Provides index-based indexing for TextTensor, leveraging the `iloc` method of the Text entries.
        You can access and manipulate sub-elements of the TextTensor based on indices.
        """

        class IlocIndexer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __getitem__(self, index):
                result = [text.iloc[index] for text in self.tensor.content.flat]
                shape = self.tensor.content.shape
                return TextTensor(np.array(result, dtype=object).reshape(shape), parse=False)

            def __setitem__(self, index, value):
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=object).reshape(self.tensor.content.shape)
                for idx, text_entry in np.ndenumerate(self.tensor.content):
                    text_entry.iloc[index] = value[idx]

        return IlocIndexer(self)

    def detach(self) -> 'TextTensor':
        with torch.no_grad():
            detached_tensor = self.clone()

        detached_tensor._is_param = False
        detached_tensor.requires_grad_(False)
        return detached_tensor

    def to(self, device, *args, **kwargs):
        """Removes the effects of the method from TextTensors,
        applies the method to embedding and/or token tensors"""
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = self.embedding.to(device, *args, **kwargs)
        if isinstance(self.tokens, torch.Tensor):
            self.tokens = self.tokens.to(device, *args, **kwargs)
        return self

    def numpy(self):
        return np.array([str(c) for c in self.content.flat]).reshape(self.content.shape)

    @classmethod
    def from_file(cls, path, encoding="utf-8", parse: Union[bool, str] = True, **kwargs):
        return cls._ttype.from_file(path, encoding=encoding, parse=parse).to_tensor(**kwargs)

    @classmethod
    def from_df(cls, df: 'DataFrame', **kwargs) -> 'TextTensor':
        text_list = []
        if not "parse" in kwargs:
            kwargs["parse"] = False

        for row in df.values:
            # Create a Text object with (column_name, value) pairs
            text_content = [Text((col_name, str(value))) for col_name, value in zip(df.columns, row)]
            text_list.append(text_content)

        # Convert the list of Text objects to a TextTensor of shape (n, 1)
        text_tensor = cls(text_list, **kwargs)

        return text_tensor

    def set_key(self, keys=None, inplace=False) -> 'TextTensor':
        assert keys is not None
        if isinstance(keys, (TextTensor, np.ndarray)):
            try:
                reshaped_keys = keys.reshape(self.content.shape)
            except ValueError:
                raise ValueError("Keys must be of the same shape as the TextTensor")
        elif isinstance(keys, str):
            reshaped_keys = utils.full_like(self.content, keys)
        elif isinstance(keys, list):
            reshaped_keys = utils.full_like(self.content, keys[0])
            for k in keys[1:]:
                reshaped_keys += k
        if inplace:
            result = self
        else:
            result = self.copy()

        # Apply set_key to each entry
        for index, text_entry in np.ndenumerate(result.content):
            result.content[index] = text_entry.set_key(reshaped_keys[index].item())

        return result

    def set_key_(self, keys) -> 'TextTensor':
        return self.set_key(keys, inplace=True)

    def add_key(self, keys, inplace=False) -> 'TextTensor':
        reshaped_keys = utils.full_like(self, keys)
        if inplace:
            result = self
        else:
            result = self.copy()
        # Apply set_key to each entry
        for index, text_entry in np.ndenumerate(result.content):
            result.content[index] = text_entry.add_key(reshaped_keys[index].item())

        return result

    def add_key_(self, keys) -> 'TextTensor':
        return self.add_key(keys, inplace=True)

    @classmethod
    def str_formatter(cls, array, indent='  '):
        """Formats a TextTensor into a matrix representation with entries padded with spaces to be able to have multi-line entries aligned.
        It would be pretty challenging to replace this in a subclass so rather, replace the Text subclass method."""
        return formatters.tensor_str_formatter(cls, array.content if isinstance(array, TextTensor) else array, indent)

    def apply(self, func):
        """Applies a function: Text -> Text to each entry of self.content"""
        # Ensure the function is callable
        if not callable(func):
            raise ValueError("Provided function is not callable.")

        # Apply the function to each entry
        for index, text_entry in np.ndenumerate(self.content):
            self.content[index] = func(text_entry)

        return self

    def inv(self):
        # Apply inverse operation on each element of the char array
        self.content = np.vectorize(lambda x: x.inv())(self.content)



    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if not hasattr(self, "_metadata"): self._metadata = {}
        # set core attributes
        attrs = ["content", "embedding", "tokens"]
        for attr in attrs:
            if attr in metadata:
                setattr(self, attr, metadata.pop(attr))

        for attr in attrs:
            metadata[attr] = getattr(self, "_" + attr)

        self._metadata = metadata

    def getitem_over_metadata(self, index):
        """Pick out entries of metadata with the provided index. Usually used to transfer metadata to a sub-tensors."""

        index = utils.positive_indices(index, self.content.shape)
        index = index[0] if len(index) == 1 else index

        if len(self.content.shape) == 0:  # and (index == slice(None, None, None))
            return self.metadata_apply(lambda v: v, lambda v: v, lambda v: v)

        else:
            return self.metadata_apply(f_tensor=lambda v: v[index],
                                       f_embedding=lambda v: v[index],
                                       f_scalar=lambda v: v)

    def metadata_apply(self, f_tensor=None, f_embedding=None, f_scalar=None, *args, **kwargs):
        """
        Apply a specified function to a copy of metadata elements of the TextTensor.

        Parameters:
        - f_tensor (function, optional): Function to be applied to tensors-shaped metadata attributes.
        - f_embedding (function, optional): Function to be applied to the 'embedding' attribute.
        - f_scalar (function, optional): Function to be applied to scalar metadata attributes.
        - *args: Variable length argument list to be passed to the functions.
        - **kwargs: Arbitrary keyword arguments to be passed to the functions.

        Returns:
        - dict: A dictionary with updated metadata attributes.
        """
        tensor_attributes, scalar_attributes = [], []
        for k, v in self._metadata.items():
            try:
                if tuple(v.shape) == tuple(self.shape):
                    tensor_attributes.append(k)
            except AttributeError:
                if k != "embedding":
                    scalar_attributes.append(k)

        metadata = self._metadata.copy()

        if f_tensor:
            for k in tensor_attributes:
                metadata[k] = f_tensor(metadata[k], *args, **kwargs)

        if f_embedding and metadata["embedding"] is not None:
            metadata["embedding"] = f_embedding(metadata["embedding"], *args, **kwargs)

        if f_scalar:
            for k in scalar_attributes:
                metadata[k] = f_scalar(metadata[k], *args, **kwargs)

        return metadata

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.str_formatter(self)

    def inspect(self):
        details = np.vectorize(lambda x: x.inspect())(self.content)
        return details

    def copy(self):
        new_dict = deepcopy(self.metadata)
        return self.__class__(metadata=new_dict, parse=False)

    clone = copy  # Alias

    def __eq__(self, other):
        return self.content == other.content if isinstance(other, TextTensor) else self.content == other

    def __ne__(self, other):
        return ~self.__eq__(other)
    def sum(self, dim: Union[int, List[int]] = None, keepdim: bool = False, sep: str = None, unique: bool = False) -> 'TextTensor':
        """Reduces the input tensor over the specified dimensions, optionally keeping dimensions."""
        if len(self.shape) == 0:
            return self

        if dim is None and sep is None:
            dim = list(range(len(self.shape)))
            connecting_char = ([" "] + ["\n" * (i + 1) for i in range(len(self.content.shape) - 1)])[::-1]
        else:
            if sep is None:
                sep = ""
            if isinstance(dim, int):
                dim = [dim]
            connecting_char = [sep] * len(dim)
        for d in dim:
            assert dim is None or d < len(
                self.shape), f"Dimension {d} is out of range for input of shape {self.content.shape}"

        result = self
        for d, char in zip(sorted(dim)[::-1], connecting_char):
            result = result.join_with(char, dim=d)
            if keepdim:
                result = result.unsqueeze(d)

        return result

    def join_with(self, sep=" ", dim=None):
        return JoinTextTensor.apply(self, sep, dim)

    join = join_with  # Alias

    def format(self, **kwargs):
        """
        Return a formatted version of self, using substitutions from kwargs.
        The substitutions are identified by braces ('{' and '}').
        """
        return FormatTextTensor.apply(self, kwargs)

    def to_list(self):
        return list(self.content.flat)

    def __pow__(self, power):
        if power == -1:
            self_copy = self.copy()
            self_copy.inv()
        else:
            raise ValueError("Only power of -1 is supported for inversion.")
        return self_copy

    def __add__(self, other):
        if isinstance(other, str) or isinstance(other, np.ndarray):
            try:
                other = TextTensor(other, parse=False)
            except ParseException:
                other = TextTensor(other, parse=False)
        return AddTextTensor.apply(self, other)

    def __mul__(self, other):
        if isinstance(other, str) or isinstance(other, np.ndarray):
            try:
                other = TextTensor(other, parse=True)
            except ParseException:
                other = TextTensor(other, parse=False)
        return MulTextTensor.apply(self, other)

    def __matmul__(self, other):
        if isinstance(other, str) or isinstance(other, np.ndarray):
            try:
                other = TextTensor(other, parse=True)
            except ParseException:
                other = TextTensor(other, parse=False)
        result = self.__class__(self.content @ other.content, parse=False)
        return result

    def item(self):
        return next(self.content.flat)

    def items(self):
        """Get (key, value) pairs from all Text entries in tensors"""

        class TextItems(tuple):
            def __init__(self, text):
                # Store the pairs as a list of tuples to allow duplicates
                self._items = text.items()

            def __hash__(self):
                return id(self)

            def __iter__(self):
                # Allow iteration over the pairs
                return iter(self._items)

            def __contains__(self, item):
                # Membership testing to check if an item is in the pairs
                return item in self._items

            def __len__(self):
                # Return the number of items
                return len(self._items)

            def __getitem__(self, item):
                return self._items[item]

            def __array__(self, *args, **kwargs ):
                return np.array(set((self,)), dtype=object)
            def __repr__(self):
                # String representation for debugging
                return f"text_items({self._items})"
        return np.array([TextItems(t) for t in self.content.flat], dtype=object).reshape(
            self.content.shape)

    def __getattr__(self, name):
        try:
            metadata = object.__getattribute__(self, "_metadata")
        except AttributeError:
            raise AttributeError(
                "Got a TextTensor without content. TextTensor was likely passed outside of langtorch, e.g. to an unsupported torch function that expects a torch.Tensor")
        if name in ["_content", "content"] or name in metadata:
            if name in metadata:
                return metadata[name]
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return None
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # Query np.ndarray attrs
            try:
                assert self.content is not None
                return getattr(self._content, name)
            except AttributeError:
                # Query Text attrs

                try:
                    attr = np.array(np.vectorize(lambda obj: getattr(obj, name))(self._content), dtype=object)
                    if callable(attr.flat[0]):
                        # alternative implementation: return lambda: np.vectorize(lambda x: x())(attr)
                        return lambda: np.array([getattr(obj, name)() for obj in self.content.flat],
                                                dtype=object).reshape(self.content.shape)
                    else:
                        return attr
                except:
                    raise AttributeError(f"Object {self}, Neither TextTensor nor Text has attribute '{name}'")

    def __getitem__(self, index):
        is_torch_return_type = lambda obj: obj.__class__.__module__ == 'torch.return_types'
        try:
            if is_torch_return_type(index):
                if "indices" in dir(index):
                    index = index.indices
                else:
                    raise ValueError(
                        "Trying to index with an unsupported torch.return_types object. Please use an index compatible with numpy indexing.")
            if isinstance(index, torch.Tensor):
                index = index.cpu().detach().numpy()
            _ = self.content[index]
        except IndexError as e:
            if ("out of bounds" in str(e)) or ("out of range" in str(e)):
                raise IndexError(f"Index {index} out of bounds for TextTensor of shape {self.content.shape}") from e
            else:
                raise e
        return self.__class__(metadata=self.getitem_over_metadata(index),
                            ttype=self.ttype,
                            embedding_model=self.embedding_model,
                            tokenizer=self.tokenizer,
                            tokenizer_kwargs=self.tokenizer_kwargs,
                            requires_grad=self.requires_grad,
                            is_gradient=self.is_gradient,
                            is_param=self.is_param,
                            parse=False)

    def __setitem__(self, index, value):
        """
        Sets the item at the specified index to value. If embeddings are present, they are invalidated for the modified entries.

        Parameters:
        - index: The index or indices specifying where to update the tensor.
        - value: The new value to set at the specified index. This can be a single Text object, a string, or a collection of them matching the index shape.
        """
        # Convert value to a compatible format, e.g., a Text object or an array of Text objects
        if not isinstance(value, (Text, np.ndarray, list)):
            raise TypeError("Value must be a Text instance, a numpy array, or a list of Text instances.")

        if isinstance(value, Text):
            value = np.array([value], dtype=object)
        elif isinstance(value, list):
            value = chararray_to_TextArray(value, self._ttype, shape=None)

        # Update the content
        if isinstance(index, torch.Tensor):
            index = index.cpu().detach().numpy()
        self.content[index] = value

        # Invalidate embeddings for the modified entries
        if self._embedding is not None:
            # Determine the shape of the embedding to be invalidated
            embedding_shape = self._embedding[index].shape
            # Create a tensor of NaN values with the appropriate shape
            nan_tensor = torch.full(embedding_shape, torch.nan, dtype=self._embedding.dtype,
                                    device=self._embedding.device)
            # Assign the NaN tensor to the slice of the embedding tensor
            self._embedding[index] = nan_tensor

        # Update metadata accordingly, if other metadata fields are affected by the item change

    def __iter__(self):
        return iter(self.content.flat)

    def __contains__(self, item):
        return item in self.content

    @property
    def TT(self):
        from ..tt import TextModule
        return TextModule(self.mT)

    @property
    def T(self):  # diff!
        return self.__class__(self.content.T, super().mT, parse=False)

    @property
    def mT(self):  # diff!
        return self.__class__(self.content.T, super().mT, parse=False)

    def embed(self, model=None, verbose=False):
        if model is None:
            model = self._embedding_model
        else:
            self.embedding_model = model
        assert model is not None
        if callable(model):
            self.embedding = model([str(m) for m in self.flat]).reshape(self.shape + (-1,))
        else:
            self.embedding = get_embedding(self, model=model, verbose=verbose)
        return self.embedding

    def apply(self, func):
        """Applies a function to each entry of self.content."""
        # Ensure the function is callable
        if not callable(func):
            raise ValueError("Provided function is not callable.")

        # Apply the function to each entry
        for index, text_entry in np.ndenumerate(self.content):
            self.content[index] = func(text_entry)

        return self

    def reshape(self, *shape):
        return ReshapeTextTensor.apply(self, shape)[0]

    def unsqueeze(tensor, dim=0):
        return tensor.reshape(tensor.shape[:dim] + (1,) + tensor.shape[dim:])

    def squeeze(tensor, dim=None):
        if dim is None:
            # Remove all dimensions of size 1
            return tensor.view([dim for dim in tensor.shape if dim != 1])
        else:
            if dim < 0:
                # Convert negative dimension to positive
                dim = len(tensor.shape) + dim
            # Remove only the specified dimension if it is of size 1
            if tensor.shape[dim] == 1:
                return tensor.view(tensor.shape[:dim] + tensor.shape[dim + 1:])
            else:
                return tensor

    def expand(tensor, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], tuple):
            sizes = sizes[0]
        # Check if the number of dimensions of the tensors is less than the length of the target sizes
        if tensor.dim() < len(sizes):
            # Add singleton dimensions to the front of the tensors
            for _ in range(len(sizes) - tensor.dim()):
                tensor = tensor.unsqueeze(0)

        # Prepare a list to hold the expanded sizes
        expanded_sizes = []

        # Loop over the target sizes from last to first
        for tensor_size, target_size in zip(reversed(tensor.shape), reversed(sizes)):
            if tensor_size == 1:
                # If the size of the tensors in this dimension is 1, it can be expanded
                expanded_sizes.append(target_size)
            elif tensor_size != target_size:
                # If the size of the tensors in this dimension is not equal to the target size,
                # and is not 1, it cannot be expanded
                raise ValueError(f"size mismatch for dimension {len(expanded_sizes)}, {tensor_size} != {target_size}")
            else:
                # If the size of the tensors in this dimension is equal to the target size,
                # it doesn't need to be expanded
                expanded_sizes.append(1)

        # Reverse the list of expanded sizes to match the original order of dimensions
        expanded_sizes = list(reversed(expanded_sizes))

        # Use the repeat method to create a new tensors that repeats the original tensors along each dimension
        return tensor.repeat(*expanded_sizes)

    def expand_as(self, other) -> 'TextTensor':
        return self.expand(other.shape)

    def swapaxes(self, axis1: int, axis2: int):
        # Create a list of axes in order
        axes = list(range(len(self.shape)))
        # Swap the specified axes
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        # Return the tensors with axes swapped
        return self.permute(axes)

    def permute(self, axes):
        return PermuteTextTensor.apply(self, axes)

    def view(self, *shape):
        return self.__class__(
            metadata=self.metadata_apply(lambda v: v.reshape(*shape), lambda v: v.reshape(*shape, v.shape[-1])),
            parse=False)

    def repeat(tensor, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], tuple):
            sizes = sizes[0]
        # Ensure the number of dimensions of tensors and sizes match
        if len(tensor.shape) != len(sizes):
            raise ValueError("Number of dimensions of tensors and sizes must match")

        # Manually create a repeated tensors
        content = tensor.content
        for dim, size in enumerate(sizes):
            slices = [content] * int(size)
            content = np.concatenate(slices, axis=dim)
        tensor = tensor.__class__(content, parse=False)
        return tensor

    def split(self, sep, dim=0):
        """
        Return a new TextTensor, with an additional first dimension to split everything using sep as the delimiter.

          sep
            The delimiter according which to split the bytearray.
            None (the default value) means split on ASCII whitespace characters
            (space, tab, return, newline, formfeed, vertical tab).
        """
        return SplitTextTensor.apply(self, sep, dim)

    def format(self, **kwargs):
        """
        Return a formatted version of self, using substitutions from args and kwargs.
        The substitutions are identified by braces ('{' and '}').
        """
        return FormatTextTensor.apply(self, kwargs)

    def save(self, filename="saved_tensors.pt"):
        torch.save(self, filename)

    def to_csv(self, filename, sep="\t"):
        """
        Save a csv with texts or, if available, with embeddings (flattened shape)
        """
        if self._embedding is None:
            self.save(filename, sep=sep)  # Tab as default, because texts have a lot of commas
        else:
            try:
                with open(filename, 'w', newline='') as f:
                    for row in self._content:
                        # Join the elements in the row with commas
                        line = sep.join(row)
                        # Write the line to the file
                        f.write(line + '\n')
            except Exception as E:
                raise Exception(f"Failed to save with embeddings, {E}")

    def backward(
            tensors: _TensorOrTensors,
            grad_tensors: Optional[_TensorOrTensors] = None,
            retain_graph: Optional[bool] = None,
            create_graph: bool = False,
            grad_variables: Optional[_TensorOrTensors] = None,
            inputs: Optional[_TensorOrTensors] = None,
            activation: Optional['Activation'] = None
    ) -> None:
        r"""See. docs"""
        if torch._C._are_functorch_transforms_active():
            raise RuntimeError(
                "backward() called inside a functorch transform. This is not "
                "supported, please use functorch.grad or functorch.vjp instead "
                "or call backward() outside of functorch transforms.")

        if grad_variables is not None:
            if grad_tensors is None:
                grad_tensors = grad_variables
            else:
                raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                                   "arguments both passed to backward(). Please only "
                                   "use 'grad_tensors'.")
        if inputs is not None and len(inputs) == 0:
            raise RuntimeError("'inputs' argument to backward() cannot be empty.")

        if grad_tensors is None:
            grad_tensors = utils.zeros_like(tensors).reshape(tensors.shape)
        elif isinstance(grad_tensors, str):
            grad_tensors = (TextTensor(grad_tensors),)

        tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)

        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else \
            tuple(inputs) if inputs is not None else tuple()
        grad_tensors_ = langtorch.torch_utils.tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
        grad_tensors_ = make_grads(tensors, grad_tensors_, is_grads_batched=False)
        if retain_graph is None:
            retain_graph = create_graph

        # Add the backward activation to grad tensors
        for t in grad_tensors_:
            if not hasattr(t, "backward_activation") or t.backward_activation is None:
                if activation is None:
                    activation = langtorch.ctx.default_model_for_backward
                assert activation is not None, "No activation provided for backward pass"
                from langtorch import Activation
                if isinstance(activation, Activation):
                    t.backward_activation = activation
                elif isinstance(activation, str):
                    t.backward_activation = Activation(activation)
                else:
                    raise ValueError("Activation must be an Activation instance or a string")

        langtorch.autograd.run_backward(  # langtorch version of a C++ engine that torch uses to run the backward pass
            tensors, grad_tensors_, retain_graph, create_graph, inputs,
            allow_unreachable=True,
            accumulate_grad=True)  # langtorch version of a C++ engine that torch uses to run the backward pass
