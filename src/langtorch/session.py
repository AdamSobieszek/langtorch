import base64
import contextvars
import hashlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass  # for storing API metadata

import torch
import yaml
from omegaconf import OmegaConf, UnsupportedValueType

from .conf import cfg_yaml_aliases


class SingletonMeta(type):
    # This dict holds a single instance of a given singleton class, indexed by the class itself
    _instance = contextvars.ContextVar('ctx',
                                       default={})

    def __call__(cls, *args, **kwargs):
        instances = cls._instance.get()  # Get the dict of single instances
        if cls not in instances:  # Add new
            instance = super().__call__(*args, **kwargs)
            instances[cls] = instance
            cls._instance.set(instances)
        else:
            if args or kwargs:  # Reload existing
                instances[cls] = instances[cls].load(*args, **kwargs)
        return instances[cls]


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


class SingletonMeta(type):
    _instance = contextvars.ContextVar('ctx', default={})

    def __call__(cls, *args, **kwargs):
        instances = cls._instance.get()  # Get the dict of single instances
        if cls not in instances:  # Add new
            instance = super().__call__(*args, **kwargs)
            instances[cls] = instance
            cls._instance.set(instances)
        else:
            if args or kwargs:  # Reload existing
                instances[cls] = instances[cls].load(*args, **kwargs)
        return instances[cls]


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


class Session(metaclass=SingletonMeta):
    """A context manager for saving and loading session data: tensors, api calls, configuration and caching"""
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf/defaults.yaml")
    session_file_template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf/new_session_template.yaml")

    def __init__(self, path=None, override=False, **kwargs):
        self._tensors = dict()
        self._requests = dict()
        self._responses = dict()
        self._override = override
        self._path = path if path is None else f"{os.path.splitext(path)[0]}.yaml"
        self._cache_file = None
        self.load(path, override, **kwargs)
        self._status_tracker = StatusTracker()

    def load(self, path=None, override=False, **kwargs):
        try:
            self._config = OmegaConf.load(self.default_config)
            if path:
                self._path = path
            if self._path and not override:
                session_config = OmegaConf.load(path)
                self._config = OmegaConf.merge(self._config, session_config)

                self._config.cache_file = f"{os.path.splitext(self._path)[0]}.pt"
                self._cache_file = self._config.cache_file
            if self._cache_file and os.path.exists(self._cache_file):
                cache = torch.load(self._cache_file)
                if "tensors" in cache:
                    self._tensors = cache.pop("tensors")
                if "responses" in cache:
                    self._responses = cache.pop("responses")
                for k, v in cache.items():
                    setattr(self, k, v)
        except Exception as e:
            logging.error(f"Error loading session configuration: {e}")
            self._config = OmegaConf.load(self.session_file_template)
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def save(self, path=None):
        if path is None:
            path = self._path

        tensors_metadata = self._config.pop('tensors', [])

        underscore_attrs = OrderedDict((k, v) for k, v in self._config.items() if k.startswith('_'))
        for k in underscore_attrs.keys():
            self._config.pop(k, None)

        self._config.tensors = tensors_metadata

        for k, v in underscore_attrs.items():
            self._config[k] = v

        with open(path, 'w', encoding="utf-8") as f:
            OmegaConf.save(self._config, f, resolve=True)

        # Save the tensor cache
        if self._cache_file:
            torch.save({
                'tensors': self._tensors,
                'responses': self._responses
            }, self._cache_file)

    def __setattr__(self, name, value, save=True):
        if name in ["_config", "_tensors", "_session_file", "_override"]:
            super().__setattr__(name, value)
            if name == "_config" and self._path and save:
                self.save()
        elif name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if isinstance(value, torch.Tensor):
                self._tensors[name] = value
            else:
                try:
                    self._config[name] = value
                    if name in self._tensors:
                        print(
                            f"Saving non-tensor with the same name as a saved tensor {name}, the tensor will be unobtainable (but remains saved)")
                except UnsupportedValueType as e:
                    raise UnsupportedValueType("Session can only hold primitive types and TextTenor objects.") from e
            if self._path and save:
                self.save()

    @property
    def path(self):
        return self._path

    def reload(self):
        return self.load()

    def request(self, provider, type, request_strings, cache=True):
        def append_spaces_to_duplicates(strings):
            occurrence_dict = {}
            result = []

            for s in strings:
                count = occurrence_dict.get(s, 0)
                occurrence_dict[s] = count + 1
                result.append(s + " " * count)

            return result

        strings = [provider + type + s for s in request_strings]
        strings = append_spaces_to_duplicates(strings)
        assert len(strings) == len(set(strings)), "Duplicate requests found"
        ids = [self.get_hash(s) for s in strings]

        if cache:
            criterion = lambda id: id not in self._responses or any([isinstance(r, dict) for r in self._responses[id]])
            ids, request_strings = [id for id in ids if criterion(id)], [request for id, request in
                                                                         zip(ids, request_strings) if
                                                                         criterion(id)]
        for id, request in zip(ids, request_strings):
            self._requests[id] = request
        return ids, request_strings

    def get_response(self, id):
        return self._responses[id] if id in self._responses else None

    def add_response(self, id, response):
        if "error" in response:
            from langtorch import Text
            response = [Text(response["error"], parse=False)]
        elif "data" in response:
            response = torch.tensor(response["data"][0]["embedding"])
            result_dict = {id: response}
            self._tensors.update(result_dict)
        elif "choices" in response:
            from langtorch import Text
            response = [Text.from_messages(m['message'], parse=False) for m in response["choices"]]
        else:
            raise ValueError(f"Invalid response: {response}")
        self._responses.update({id: response})

    def prompts(self, ids):
        return [self._requests[id] if id in self._requests else None for id in ids]

    def completions(self, type, provider, request_strings, key=None):
        if len(request_strings) == 0:
            return []

        ids, _ = self.request(provider, type, request_strings, False)
        responses = [self.get_response(id) for id in ids]
        for i, response in enumerate(responses):
            if response is None:
                raise ValueError(f"Could not find response for {request_strings[i]}")


        if type == "embeddings":
            return torch.vstack(responses)  # .permute((1,0))
        else:
            for i, response in enumerate(responses):
                for j, text in enumerate(response):
                    if key is None and len(text.items())==1 and text.items()[0][0] == "assistant":
                        responses[i][j].content = [text.items()[0][1]]
            from langtorch import TextTensor
            # Here, sublists represent choices for different prompts and sublist elements are Text objects
            responses = [
                [text.add_key(key) if (
                            key is not None and len(text.items()) != 0 and text.items()[0][0] != key) else text for
                 text in response] for
                response in responses]
            return TextTensor(responses, parse=False)  # .permute((1, 0))

    def __setattr__(self, name, value, save=True):
        if name in ["_config", "_tensors", "_session_file", "_override"]:
            super().__setattr__(name, value)
            if name == "_config" and self._path and save:
                self.save()
        elif name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if isinstance(value, torch.Tensor):
                self._tensors[name] = value
            else:
                try:
                    self._config[name] = value
                    if name in self._tensors:
                        print(
                            f"Saving non-tensor with the same name as a saved tensor {name}, the tensor will be unobtainable (but remains saved)")
                except UnsupportedValueType as e:
                    raise UnsupportedValueType("Session can only hold primitive types and TextTenor objects.") from e
            if self._path and save:
                self.save()

    @property
    def tensors(self):
        return self._tensors

    @property
    def config(self):
        return self._config

    @property
    def cfg(self):
        return self._config

    def __getattr__(self, name):
        if name in ["_config", "_session_file", "_cache_file", "_tensors", "_override"]:
            return super().__getattribute__(name)

        if not hasattr(self._config, name) and name in self._tensors:
            return self._tensors[name]

        try:
            if hasattr(self._config, name):
                attr = self._config[name]
            elif hasattr(self._config.cfg, name):
                attr = self._config.cfg[name]
            elif hasattr(self._config.methods, name):
                attr = self._config.methods[name]
            else:
                raise KeyError()
        except KeyError as e:
            logging.warning(f"getattr {name}, but session attributes are: {self._config.keys()}")

            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from e
        return attr

    def get_hash(self, *args):
        combined_string = ''.join(map(str, args))
        hex_string = hashlib.sha256(combined_string.encode()).hexdigest()[0:32]
        base64_encoded = base64.b64encode(bytes.fromhex(hex_string))
        return base64_encoded.decode('utf-8').replace("/", "_")

    @classmethod
    def create_aliases(cls, aliases):
        for alias, path in aliases.items():
            def create_property(path):
                def property_func(self):
                    return getattr(self, path)

                return property(property_func)

            setattr(cls, alias, create_property(path))

    def __getitem__(self, entry):
        return self.__getattr__(entry)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return "---   Session Config   ---\n" + yaml.dump(
            OmegaConf.to_object(self._config)) + "---   --------------   ---\n"

    def __repr__(self):
        return self.__str__()

    def _delete(self):
        os.remove(self._path)
        Session.current_session.set({})

    def get_tensor_metadata(self):
        raise NotImplementedError

    # Make Session a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._path:  # Save to file if a path is set
            self.save()
        # If any exception occurs, you can handle it here.
        if exc_type is not None and not exc_type is SystemExit:
            logging.error(f"An error occurred: {exc_val.__class__}")
        return False  # False means don't suppress exceptions


ctx = Session()
ctx.create_aliases(cfg_yaml_aliases)
context = ctx
