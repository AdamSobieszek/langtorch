"""
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (abstract; API interface for inputs, outputs, metadata; one method to call API)
        - APIRequestOpenAI( ... + tokens_consumed)
        - APIRequestGCP ( ... + letters_consumed)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - task_id_generator_function (yields 1, 2, 3, ...)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - num_letters_consumed_from_request

"""
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit
from abc import ABC, abstractmethod  # for abstract APIRequest class
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata

import aiohttp  # for making API calls concurrently
from aiohttp import ClientConnectorError
import tiktoken  # for counting tokens

from ..session import ctx as langtorch_session


# dataclasses

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


@dataclass
class APIRequest(ABC):
    """Abstract API request class."""
    task_id: int
    id: str
    request_json: dict
    attempts_left: int
    result = []

    @abstractmethod
    async def call_api(self, *args, **kwargs):
        """Required method for calling the API."""
        pass


@dataclass
class APIRequestOpenAI(APIRequest):
    """Represents an OpenAI API request."""
    token_consumption: int

    async def call_api(
            self,
            request_url: str,
            request_header: dict,
            retry_queue: asyncio.Queue,
            status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json(content_type='application/json')
            if "error" in response:
                logging.debug(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately
        except ClientConnectorError as e:
            raise e  # Re-raise a connection error
        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                errors = '\n'.join(list(set([str(r['error']) for r in self.result])))
                logging.error(f"Request {self.request_json} failed after all attempts. Encountered errors: {errors}")

                try:  # error catching to deal with session errors
                    langtorch_session.add_response(self.id, response)
                except Exception as e:
                    logging.warning(f"Failed to add {self.task_id} API ERROR response, caused by {e}")
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            try:
                langtorch_session.add_response(self.id, response)
            except Exception as e:
                logging.warning(f"Failed to add {self.task_id} API response, caused by {e}")
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to responses")


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""

    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    if match is None:
        return None
    else:
        return match[1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json_string + "\n")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def num_tokens_consumed_from_request(
        request_json: dict,
        api_endpoint: str,
        token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def num_letters_consumed_from_request(request_json: dict, api_endpoint: str) -> int:
    """Count the number of letters in the request. Note that this is a naive implementation
    and might not reflect the actual cost of a request on GCP's Vertex AI."""

    # If the request is for a completion-like endpoint
    if api_endpoint.endswith("completions"):
        # For this example, we're considering the 'prompt' key in the request JSON
        # You might need to adjust this based on GCP's actual request format
        prompt = request_json["prompt"]

        if isinstance(prompt, str):  # single prompt
            return len(prompt)
        elif isinstance(prompt, list):  # multiple prompts
            return sum([len(p) for p in prompt])
        else:
            raise TypeError('Expecting either string or list of strings for "prompt" field in the request')

    # If the request is for an embedding-like endpoint
    elif api_endpoint == "embeddings":
        input_text = request_json["input"]

        if isinstance(input_text, str):  # single input
            return len(input_text)
        elif isinstance(input_text, list):  # multiple inputs
            return sum([len(i) for i in input_text])
        else:
            raise TypeError('Expecting either string or list of strings for "input" field in the request')

    # More logic can be added here to handle other types of GCP requests
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')
