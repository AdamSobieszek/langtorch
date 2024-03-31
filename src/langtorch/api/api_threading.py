import logging

from .openai_parallel_processor import process_api_requests_from_session


async def execute_api_requests_in_parallel(
        ids: list,
        request_strings: list,
        request_url: str,
        api_key: str,
        max_requests_per_minute: float = 3_000 * 0.5,
        max_tokens_per_minute: float = 250_000 * 0.5,
        token_encoding_name: str = "cl100k_base",
        max_attempts: int = 3,
        logging_level: int = logging.ERROR,
):
    # run tasks in parallel
    try:
        await process_api_requests_from_session(
            ids=ids,
            request_strings=request_strings,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    except Exception as e:
        logging.error(f"Exception in execute_api_requests_in_parallel: {e}")
