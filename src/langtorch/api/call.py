import asyncio
import json
import logging
import os
from typing import List

import nest_asyncio
import openai

try:
    nest_asyncio.apply()
except RuntimeError:
    def get_or_create_eventloop():
        try:
            return asyncio.get_event_loop()
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return asyncio.get_event_loop()


    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()

from ..session import Session
from .api_threading import execute_api_requests_in_parallel
from .tools import generate_schema_from_function
from .utils import override


def auth(key_or_path=None):
    """
    Authenticates with the OpenAI API by setting the API key either directly or by reading from a file.

    Parameters:
        key_or_path (str, optional): Directly provided API key for authentication or path to the file containing the API key.

    Raises:
        Exception: If the key file cannot be read or does not contain valid authentication information.
    """

    # Check if key_or_path is a file path
    is_file_path = False
    if key_or_path is not None:
        is_file_path = os.path.isfile(key_or_path)

    if is_file_path:
        # If it's a file path, read the keys from the file
        try:
            with open(key_or_path) as f:
                keys = json.load(f)
                openai.api_key = keys["openai_api_key"]
                openai.organization = keys["openai_organization"]
                os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]
                os.environ["OPENAI_ORGANIZATION"] = keys["openai_organization"]
        except:
            raise Exception("Authentication failed: Unable to read or parse the key file.")
    elif key_or_path is not None:
        # If it's not a file path, assume it's an API key
        os.environ["OPENAI_API_KEY"] = key_or_path
        openai.api_key = key_or_path
    else:
        # If key_or_path is None, try to read from a default path
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'api_keys.json')
        try:
            with open(default_path) as f:
                keys = json.load(f)
                openai.api_key = keys["openai_api_key"]
                openai.organization = keys["openai_organization"]
                os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]
                os.environ["OPENAI_ORGANIZATION"] = keys["openai_organization"]
        except:
            raise Exception("Authentication failed: Unable to read or parse the default key file.")


def call(prompt, system_message, model="gpt-3.5-turbo-0613", as_str=False):
    """
    Sending a single system message prompt to get a response.

    Parameters:
        prompt (str): The input message to send to the model.
        system_message (str): Message that sets the behavior of the model.
        model (str, optional): The specific model to be used. Default is "gpt-3.5-turbo-0613".
        as_str (bool, optional): If True, returns just the 'choices' texts from the response. If False, returns the entire response object. Default is False.

    Returns:
        Union[openai.ChatCompletion, str]: The chat response object, or a string if as_str is True.
    """
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}] if isinstance(
        prompt, str) else [{"role": "system", "content": system_message}] + [{"role": r, "content": c} for r, c in
                                                                             prompt]
    chat = openai.OpenAI().chat.completions.create(model=model, messages=messages)
    return chat if not as_str else chat.choices[0]['message']['content']


def chat_strings(prompts, system_messages, model="gpt-3.5-turbo-0613", temperature=1, top_p=1, n=1,
                 stop=None, max_tokens=None, presence_penalty=0, frequency_penalty=0,
                 tools=None, tool_choice="none", **kwargs):
    """
        Prepares a list of chat strings in JSON format for batch processing. Each string represents a separate chat prompt for the OpenAI API.

        Parameters:
            prompts (List[str] or List[Tuple[str,str]]): List of prompts or (role, prompt) pairs.
            system_messages (List[str]): List of system messages to guide the model's behavior.
            model (str, optional): Model name to use. Default is "gpt-3.5-turbo-0613".
            temperature (float, optional): Sampling temperature. Default is 1.
            top_p (float, optional): Nucleus sampling parameter. Default is 1.
            n (int, optional): Number of chat completion choices to generate. Default is 1.
            stop (str or List[str], optional): Up to 4 sequences where the API will stop generating further tokens.
            max_tokens (int, optional): Maximum number of tokens to generate.
            presence_penalty (float, optional): Penalty for new tokens based on their presence so far.
            frequency_penalty (float, optional): Penalty for new tokens based on their frequency so far.
            tools (List[dict], optional): List of tools defined in a JSON schema.
            tool_choice (str, optional): The type of function call to make. Default is "none".
            **kwargs: Ignored additional parameters.

        Returns:
            List[str]: A list of request strings in JSON format.
        """

    if tools is not None:
        for i, tool in enumerate(tools):
            if isinstance(tool, str):
                try:
                    tools[i] = json.loads(tool)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Tool {tool} is not a valid JSON string object.") from e
            if not "type" in tools[i]:
                tools[i] = {"type": "function", "function": tools[i]}

    params = {"temperature": temperature,
                  "top_p": top_p,
                  "n": n,
                  "stop": stop,
                  "max_tokens": max_tokens,
                  "presence_penalty": presence_penalty,
                  "frequency_penalty": frequency_penalty,
                  "tools": tools,
                  "tool_choice": tool_choice}

    default_values = {"temperature": 1, "top_p": 1, "n": 1, "stop": None, "max_tokens": None, "presence_penalty": 0,
                      "frequency_penalty": 0, "tools": None, "tool_choice": "none"}

    jobs = [{"model": model,
             "messages": ([{"role": "system", "content": system_message}] if system_message else [])+([{"role": "user", "content": prompt}] if isinstance(prompt, str) else
             [{"role": r, "content": c} for r, c in prompt]),
             **{param: value for param, value in params.items() if value != default_values[param]}}
            for prompt, system_message in zip(prompts, system_messages)]
    return [json.dumps(job, ensure_ascii=False) for job in jobs]


def chat(prompts, system_messages, model="gpt-3.5-turbo", provider = "openai", cache=True, api_key=None, verbose=False, as_str=False,
         **kwargs):
    """
    Processes a list of chat prompts in parallel, saving the results to a specified file.

    Parameters:
        prompts (List[str] or List[Tuple[str,str]] or str): List or single prompt or list of (role, prompt) pairs to send to the model.
        system_messages (List[str] or str): List or single system message to guide the model's behavior.
        model (str, optional): Model name to use. Default is "gpt-3.5-turbo-0613".
        provider (str, optional):  The provider to use. Default is "openai".
        cache (bool, optional): If True, reuse previous call results.
        api_key (str, optional): API key for authentication. If None, uses environment variable.
        verbose (bool, optional): If True, enables detailed logging. Default is True.
        as_str (bool, optional): If True, returns only the model's message as string, otherwise returns the entire response object. Default is False.
        **kwargs: Additional parameters  of the api request.

    Returns:
        Union[Coroutine, openai.ChatCompletion, str]: Coroutine object representing the asynchronous execution of the API requests if save_filepath is provided. Otherwise, returns the chat response object, or a string if as_str is True.
    """

    if "T" in kwargs:
        kwargs["temperature"] = kwargs["T"]
    if not isinstance(system_messages, list): system_messages = [system_messages]
    if not isinstance(prompts, list):
        prompts = [prompts]
    if len(system_messages) == 1:
        system_messages = system_messages * len(prompts)
    request_strings = chat_strings(prompts, system_messages, model, **kwargs)

    session = Session()
    provider_to_request_url = {"openai":"https://api.openai.com/v1/chat/completions",
                                               "anthropic":"https://api.anthropic.com/v1/messages",
                                               "groq":"https://api.groq.com/openai/v1/chat/completions"}
    assert provider in provider_to_request_url, f"Provider {provider} not in available providers"
    if api_key is None:
        try:
            api_key = os.environ[f"{provider.upper()}_API_KEY"]
        except KeyError as e:
            raise KeyError(f'No {provider} API key. Set os.environ["{provider.upper()}_API_KEY"] = YOUR KEY') from e
    provider_to_request_header = {"openai": {"Authorization": f"Bearer {api_key}"},
                                 "anthropic": {"x-api-key:": f"{api_key}",
                                               "anthropic-version": "2023-06-01",
                                               "Content-Type": "application/json"},
                                      "groq": {"Authorization": f"Bearer {api_key}"},}
    request_header = provider_to_request_header[provider]
    ids, uncached_request_strings = session.request(provider, "chat", request_strings, cache)

    with Session(session.path) as session:
        if request_strings:
            job = execute_api_requests_in_parallel(
                ids=ids,
                request_strings=uncached_request_strings,
                request_url=provider_to_request_url[provider],
                request_header=request_header,
                logging_level=logging.INFO if verbose else logging.ERROR,
                max_requests_per_minute=200 if "gpt-4" in request_strings[0] else 3_500 * 0.5,
                max_tokens_per_minute=40_000 * 0.5 if "gpt-4" in request_strings[0] else 90_000 * 0.5,
            )

            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(job)
            except RuntimeError as E1:
                try:
                    asyncio.run(job)
                except RuntimeError as E2:
                    raise RuntimeError(
                        f"Asyncio get current event loop failed with {E1}. Tried to fall back on new event loop with asyncio.run(job) but failed with error: {E2}.\nThis indicates error with the API code or provider and could be hard to fix.")

        return session.completions("chat", provider, request_strings)


def get_embedding(texts, model="text-embedding-3-small", cache=True,  api_key=None, verbose=False):
    """
    Retrieves embeddings for the given texts from the OpenAI API and saves the results in a file.

    Parameters:
        texts (List[str], TextTensor): List of texts for which to get embeddings.
        cache (bool, optional): If True, reuse previous call results. Default is True.
        api_key (str, optional): API key for authentication. If None, uses environment variable.
        verbose (bool, optional): If True, enables detailed logging. Default is True.

    Returns:
        dict: The embeddings from the saved results file, loaded as a Python dictionary.
    """
    from langtorch import TextTensor
    session = Session()

    if api_key is None:
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError('No OpenAI API key. Set os.environ["OPENAI_API_KEY"] = YOUR KEY')
    if isinstance(texts, str): texts = [str(texts)]
    if isinstance(texts, TextTensor):
        shape = texts.content.shape
        texts = [str(m) for m in texts.content.flat]
    else:
        shape = (len(texts),)
    request_strings = [json.dumps({"model": model, "input": str(x).strip() + "\n"}, ensure_ascii=False) for x in texts]

    ids, uncached_request_strings = session.request("openai", "embeddings", request_strings, cache)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # Execute API requests in parallel and save to session
    with Session(session.path) as session:
        job = execute_api_requests_in_parallel(
            ids=ids,
            request_strings=uncached_request_strings,
            request_url="https://api.openai.com/v1/embeddings",
            request_header=request_header,
            max_requests_per_minute=3_000 * 0.5,
            max_tokens_per_minute=250_000 * 0.5,
            token_encoding_name="cl100k_base",
            max_attempts=3,
            logging_level=logging.INFO if verbose else logging.ERROR,
        )

        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(job)
        except Exception as E:
            asyncio.run(job)

        return session.completions("embeddings", "openai", request_strings).reshape(shape + (-1,))


embed = get_embedding  ## Alternative function name
