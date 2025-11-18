from requests.exceptions import HTTPError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

MAX_RETRIES = 3


def is_retriable_error(exception):
    """Check if the error should trigger a retry."""
    if isinstance(exception, HTTPError):
        return exception.response.status_code in [409, 429, 500, 502, 503, 504]
    return False


@retry(
    retry=retry_if_exception(is_retriable_error),
    wait=wait_exponential(multiplier=1, min=0.5, max=5),
    stop=stop_after_attempt(MAX_RETRIES),
)
def call_api_with_retry(search_fn, *args, **kwargs):
    return search_fn(*args, **kwargs)
