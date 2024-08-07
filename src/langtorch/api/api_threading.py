import logging

from .openai import process_api_requests_from_session


async def execute_api_requests_in_parallel(
        ids: list,
        request_strings: list,
        request_url: str,
        request_header: dict,
        max_requests_per_minute: float = 5_000,
        max_tokens_per_minute: float = 250_000,
        token_encoding_name: str = "cl100k_base",
        max_attempts: int = 3
):
    # run tasks in parallel
    try:
        await process_api_requests_from_session(
            ids=ids,
            request_strings=request_strings,
            request_url=request_url,
            request_header=request_header,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts)
        )
    except Exception as e:
        logging.error(f"Exception in executing api requests in parallel: {e}")
