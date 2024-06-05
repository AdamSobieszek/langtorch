import asyncio  # for running API calls concurrently
import json
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit

from .parallel_processor_utils import APIRequestOpenAI as APIRequest
from .parallel_processor_utils import StatusTracker
from .parallel_processor_utils import api_endpoint_from_url, num_tokens_consumed_from_request, \
    task_id_generator_function
from ..session import Session


async def process_api_requests_from_session(
        ids: list,
        request_strings: list,
        request_url: str,
        request_header: dict,
        max_requests_per_minute: float,
        max_tokens_per_minute: float,
        token_encoding_name: str,
        max_attempts: int,
        logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging and session
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")
    session = Session()

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = session._status_tracker  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    _num_tasks_failed = status_tracker.num_tasks_failed
    _num_tasks_started= status_tracker.num_tasks_started
    _num_rate_limit_errors = status_tracker.num_rate_limit_errors

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")
    # initialize file reading
    requests = [(id, json.loads(m)) for id, m in zip(ids, request_strings)].__iter__()
    # `requests` will provide requests one at a time
    logging.debug(f"File opened. Entering main loop")
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    id, request_json = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        id=id,
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint,
                                                                           token_encoding_name)  if "openai.com" in request_url else 0,
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to session""")
    failed, completed = status_tracker.num_tasks_failed - _num_tasks_failed, status_tracker.num_tasks_started - _num_tasks_started
    exceeded = status_tracker.num_rate_limit_errors - _num_rate_limit_errors
    if failed > 0:
        logging.warning(
            f"{failed} / {completed} requests failed. Errors logged to session.")
    if _num_rate_limit_errors > 0:
        logging.warning(
            f"{_num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")
