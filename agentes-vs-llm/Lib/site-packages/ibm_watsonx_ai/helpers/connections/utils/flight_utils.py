#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import functools
import logging
import random
import time
from typing import Callable, Protocol

from pyarrow import flight

from ibm_watsonx_ai.wml_client_error import WMLClientError


class CallbackSchema(Protocol):
    logger: logging.Logger

    def status_message(self, msg: str) -> None: ...


class SimplyCallback:
    def __init__(self, *, logger: logging.Logger) -> None:
        self.logger = logger

    def status_message(self, msg: str) -> None:
        self.logger.debug(msg)


class HeaderMiddleware(flight.ClientMiddleware):
    def __init__(self, *, headers: dict) -> None:
        super().__init__()
        self.headers = headers

    def sending_headers(self) -> dict:
        authorization_header = self.headers.get("Authorization")
        if not authorization_header or not (
            authorization_header.startswith(("Bearer", "Basic"))
        ):
            raise WMLClientError(
                "The authorization header is missing or does not contain supported token type. Allowed token types: Bearer, Basic"
            )

        headers = {"Authorization": authorization_header}

        if impersonate_header := self.headers.get("impersonate"):
            headers.update({"Impersonate": impersonate_header})

        return headers


class HeaderMiddlewareFactory(flight.ClientMiddlewareFactory):
    def __init__(self, *, headers: dict):
        self.headers = headers

    def start_call(self, info: flight.CallInfo) -> flight.ClientMiddleware:
        return HeaderMiddleware(headers=self.headers)


def _flight_retry(max_retries: int = 3):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            attempt = 0
            while attempt < max_retries + 1:
                prolong = 0
                try:
                    return func(self, *args, **kwargs)

                except flight.FlightUnauthenticatedError as e:
                    last_exception = e
                    # suggest CPD users to check the Flight variables
                    if hasattr(self._api_client, "ICP_PLATFORM_SPACES"):
                        if self._api_client.ICP_PLATFORM_SPACES:
                            raise ConnectionError(
                                "Cannot connect to the Flight service. Please make sure you set correct "
                                "FLIGHT_SERVICE_LOCATION and FLIGHT_SERVICE_PORT environmental variables.\n"
                                "If you are trying to connect to FIPS-enabled cluster, "
                                "please set the following as environment variable and try again:\n"
                                "GRPC_SSL_CIPHER_SUITES="
                                "ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384 "
                                f"ECDHError: {e}"
                            ) from e
                    else:
                        raise ConnectionError(
                            f"Cannot connect to the Flight service. Error: {e}"
                        ) from e

                except flight.FlightUnavailableError as e:
                    last_exception = e
                    if "failed to connect to all addresses" in str(e):
                        self._logger.debug(
                            "Cannot connect to Flight Service. "
                            "Flight Service can be restarting, prolongation sleeping time to 60s"
                        )
                        prolong = 60
                    else:
                        raise

                self._logger.debug(f"Retry {attempt + 1} of {max_retries} in progress.")

                jitter = 1 + 0.25 * random.random()
                sleep_seconds = min(
                    pow(2.0, attempt) * jitter + prolong, self._max_retry_time
                )
                time.sleep(sleep_seconds)
                attempt += 1

            self._logger.debug(f"Maximum retries reached: {attempt}/{max_retries}.")
            raise last_exception

        return wrapper

    return decorator
