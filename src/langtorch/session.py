import base64
import contextvars
import datetime
import hashlib
import logging
import os
from collections import OrderedDict

import h5py
# time the execution
import torch
import yaml
from omegaconf import OmegaConf
from omegaconf.errors import UnsupportedValueType
from dataclasses import dataclass  # for storing API metadata

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


class Session(metaclass=SingletonMeta):
    """A context manager for saving and loading session data: tensors, api calls, configuration and caching"""
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf/defaults.yaml")
    session_file_template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf/new_session_template.yaml")

    def __init__(self, session_file=None, cache_file=None, new_session_file=False):
        self._tensors = dict()
        self._requests = dict()
        self._responses = dict()
        self._overwrite = new_session_file
        self._session_file = session_file
        self._cache_file = None if cache_file is None else cache_file if "." in cache_file else cache_file + ".hdf5"
        self.load(session_file, cache_file, new_session_file)
        self.status_tracker = StatusTracker()
        assert self.tensor_savepath is not None

    def load(self, path=None, cache=None, new_session_file=False):
        """Called to set or change the path and load or reload the config"""
        _config = OmegaConf.load(self.default_config)
        _config["tensors"] = getattr(_config, "tensors", [])

        self._session_file = path if path else self._session_file
        self._cache_file = cache if cache else self._cache_file

        if self._cache_file and os.path.exists(self._cache_file):
            import numpy as np
            with h5py.File(self._cache_file, 'r') as f:
                for key in f.keys():
                    if key not in self._responses:
                        dataset = f[key]
                        self._responses[key] = [m.decode('utf-8') for m in
                                                dataset[:]] if dataset.dtype == np.object_ else torch.from_numpy(
                            dataset[:])

        if self._session_file and os.path.exists(self._session_file) and not new_session_file:
            try:
                _config = OmegaConf.merge(_config, OmegaConf.load(self._session_file))
            except Exception as e:
                print(f"Error loading session from {self._session_file}: {e}")

        if not getattr(_config, "tensor_savepath", None):
            _config.tensor_savepath = "" if self._session_file is None else os.path.join(
                os.path.dirname(os.path.abspath(self._session_file)), "saved_tensors.pt")

        if _config.tensor_savepath and os.path.exists(_config.tensor_savepath):
            self._tensors = torch.load(_config.tensor_savepath)

        self._config = _config
        return self

    @property
    def path(self):
        return self._session_file

    def reload(self):
        """Called to ensure config is up-to-date"""
        return self.load()

    def request(self, provider, type, request_strings, cache=True):
        """Query the session for cached result or add new requests"""

        def append_spaces_to_duplicates(strings):
            occurrence_dict = {}
            result = []

            for s in strings:
                count = occurrence_dict.get(s, 0)
                occurrence_dict[s] = count + 1
                result.append(s + " " * count)

            return result

        # Generate unique hashes from the request strings
        strings = [provider + type + s for s in request_strings]
        strings = append_spaces_to_duplicates(strings)
        assert len(strings) == len(set(strings)), "Duplicate requests found"
        ids = [self.get_hash(s) for s in strings]

        # for id, request in zip(ids, request_strings):
        #     if not '"user"' in request and not '"assistant"' in request and id not in self._responses:
        #         from langtorch import Text
        #         logging.debug(f"Activation got request without a 'user' or 'assistant' message: {request}\nEmpty response set by default")
        #         self._responses[id]  = [Text()]
        if cache:
            ids, request_strings = [id for id in ids if not id in self._responses], [request for id, request in
                                                                                     zip(ids, request_strings) if
                                                                                     not id in self._responses]

        for id, request in zip(ids, request_strings):
            self._requests[id] = request
        return ids, request_strings

    def get_response(self, id):
        """Retrieve a cached response"""
        return self._responses[id] if id in self._responses else None

    def add_response(self, id, response):
        """Append an api response payload to response list."""
        if "error" in response:
            from langtorch import Text
            response = [Text(response["error"], parse=False)]
        elif "data" in response:
            response = torch.tensor(response["data"][0]["embedding"])
            if self._cache_file:
                self._append_tensor_to_cache(id, response)
        elif "choices" in response:
            from langtorch import Text
            response = [Text.from_messages(m['message'], parse=False) for m in response["choices"]]
            if self._cache_file:
                self._append_chat_to_cache(id, response)
        else:
            raise ValueError(f"Invalid response: {response}")
        self._responses.update({id: response})

    def _append_tensor_to_cache(self, key, tensor_data):
        with h5py.File(self._cache_file, 'a') as f:
            if key in f:
                del f[key]  # Delete the existing dataset
            f.create_dataset(key, data=tensor_data.numpy())
        ### Alternative, slower implementation with torch.save
        #     import torch
        #     result_dict = dict([*zip(inputs, embeddings)])
        #     if self.tensor_savepath:
        #         if not os.path.exists(self.tensor_savepath):
        #             torch.save(result_dict, self.tensor_savepath)
        #         else:
        #             tensors = torch.load(self.tensor_savepath)
        #             assert isinstance(tensors, dict)
        #             tensors.update(result_dict)
        #             torch.save(tensors, self.tensor_savepath)
        #     else:
        #         self._tensors.update(result_dict)
        # self.save()

    def _append_chat_to_cache(self, key, texts: list):
        assert isinstance(texts, list)
        with h5py.File(self._cache_file, 'a') as f:
            if key in f:
                del f[key]  # Delete the existing dataset
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(key, (len(texts),), dtype=dt)
            for i, text in enumerate(texts):
                f[key][i] = str(text)

    def prompts(self, ids):
        return [self._requests[id] if id in self._requests else None for id in ids]

    def completions(self, type, provider, request_strings):
        if len(request_strings) == 0:
            return []

        ids, _ = self.request(provider, type, request_strings, False)
        responses = [self.get_response(id) for id in ids]
        for i, response in enumerate(responses):
            if response is None:
                raise ValueError(f"Could not find response for {request_strings[i]}")

        if type == "embeddings":
            return torch.vstack(responses)
        else:
            from langtorch import TextTensor
            responses = [
                [text.add_key("assistant") if len(text.items()) != 0 and text.items()[0][0] != "assistant" else text for
                 text in response] for
                response in responses]
            return TextTensor(responses, parse=False)

    def save(self, path=None):
        """Save the current configuration to a file"""
        if path is None:
            path = self._session_file
        if path is None:
            raise ValueError("No session file path provided")
        # Decompose the current configuration
        tensor_savepath = self._config.pop('tensor_savepath', None)
        tensors_metadata = self._config.pop('tensors', [])

        # Filter attributes starting with an underscore
        underscore_attrs = OrderedDict((k, v) for k, v in self._config.items() if k.startswith('_'))
        for k in underscore_attrs.keys():
            self._config.pop(k, None)

        self._config['tensor_savepath'] = tensor_savepath
        self._config.tensors = tensors_metadata

        # Merge underscore attributes at the end
        for k, v in underscore_attrs.items():
            self._config[k] = v

        with open(self._session_file, 'w', encoding="utf-8") as f:
            OmegaConf.save(self._config, f, resolve=True)

    def __setattr__(self, name, value, save=True):
        if name in ["_config", "_tensors", "_session_file", "_overwrite"]:
            # Use the base class's __setattr__ to prevent recursive calls
            super().__setattr__(name, value)
            if name == "_config" and self._session_file and save:
                self.save()
        elif name.startswith('_'):
            # Set the attribute directly
            super().__setattr__(name, value)
        else:
            if isinstance(value, torch.Tensor):
                tensor_savepath = self._config.tensor_savepath
                tensors_metadata = self._config.pop('tensors', [])
                timestamp = datetime.datetime.now().isoformat()
                if not os.path.exists(tensor_savepath):
                    tensors = torch.load(tensor_savepath)
                    tensors[name] = value
                    torch.save(tensors, tensor_savepath)
                else:
                    torch.save({name: value}, tensor_savepath)

                metadata = {
                    "id": name,
                    "object": str(type(value)),
                    "created": timestamp,
                    "shape": tuple(value.shape)
                }
                if name in [m["id"] for m in tensors_metadata]:
                    tensors_metadata = [m for m in tensors_metadata if m["id"] != name]
                self._config["tensors"] = list(tensors_metadata) + [metadata]
            else:
                try:
                    self._config[name] = value
                    if name in [m["id"] for m in self._config["tensors"]]:
                        print(
                            f"Saving non-tensor with the same name as a saved tensor {name}, the tensor will be unobtainable (but remains saved)")
                except UnsupportedValueType:
                    raise UnsupportedValueType("Session can only hold primitive types and TextTenor objects.")
            if self._session_file and save:
                self.save()

    @property
    def tensors(self):
        if self.tensor_savepath:
            tensors = torch.load(self._config["tensor_savepath"])
        else:
            tensors = self._tensors
        return tensors

    def __getattr__(self, name):
        if name in ["_config", "_session_file", "_cache_file", "_tensors", "_overwrite"]:
            return super().__getattribute__(name)
        try:
            # This block of code seems intended to ensure that 'tensors' attribute exists
            # and initializes it if it doesn't, which should probably be handled elsewhere.
            _ = (self._config.tensors)
        except Exception:
            self.tensors = []

        # If the attribute name corresponds to an id in the tensors list, return that tensors
        if not hasattr(self._config, name) and name in [m["id"] for m in self._config["tensors"]]:
            return self.tensors[name]

        # Return the attribute from the configuration
        try:
            attr = self._config[name]
        except KeyError as e:
            logging.warning(f"getattr {name}, but session attributes are: {self._config.keys()}")

            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

        return attr

    def get_hash(self, *args):
        """
        Generates hash converted to base64 from a list of string requests.
        This serves as a unique ID for an api request for caching purposes.

        :return: A base64 SHA-256 hash
        """
        combined_string = ''.join(map(str, args))
        # Generate a SHA-256 hash
        hex_string = hashlib.sha256(combined_string.encode()).hexdigest()[0:32]
        # Convert to base 64
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

    def _delete(self):
        # Cleanup: remove the session file
        os.remove(self._session_file)
        Session.current_session.set({})

    def get_tensor_metadata(self):
        raise NotImplementedError


ctx = Session()
ctx.create_aliases(cfg_yaml_aliases)
