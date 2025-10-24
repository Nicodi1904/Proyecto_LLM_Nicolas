#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

import httpx
import requests

from ibm_watsonx_ai.utils.auth.base_auth import (
    STATUS_FORCELIST,
    RefreshableTokenAuth,
    TokenInfo,
    _get_token_info,
)
from ibm_watsonx_ai.utils.utils import _requests_retry_session
from ibm_watsonx_ai.wml_client_error import (
    AuthenticationError,
    InvalidCredentialsError,
    WMLClientError,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class IAMTokenAuth(RefreshableTokenAuth):
    """IAM token authentication method class.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None = None,
        on_token_refresh: Callable[[], None] | None = None,
    ) -> None:
        RefreshableTokenAuth.__init__(
            self, api_client, on_token_creation, on_token_refresh
        )

        if not api_client._is_IAM():
            raise WMLClientError(
                "api_key for IAM token is not provided in credentials for the client."
            )

    def _get_token_request_arguments(self) -> tuple[str, str, dict[str, str]]:
        url = self._api_client._href_definitions.get_iam_token_url()

        data = "apikey=" + self._api_client._href_definitions.get_iam_token_api(
            self._api_client.credentials.api_key
        )

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic Yng6Yng=",
        }

        return url, data, headers

    def _handle_token_response(
        self, response: requests.Response | httpx.Response
    ) -> TokenInfo:
        if response.status_code == 200:
            return TokenInfo(response.json().get("access_token"))
        if 400 <= response.status_code < 500:
            raise InvalidCredentialsError(reason=response.text)

        raise AuthenticationError("IAM", response)

    def _generate_token(self) -> TokenInfo:
        """Generate token using IAM authentication.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

        url, data, headers = self._get_token_request_arguments()

        response = _requests_retry_session(status_forcelist=STATUS_FORCELIST).post(
            url=url, data=data, headers=headers
        )

        return self._handle_token_response(response)

    async def _agenerate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

        url, data, headers = self._get_token_request_arguments()

        response = await self._api_client.async_httpx_client.post(
            url=url, data=data, headers=headers
        )

        return self._handle_token_response(response)


def get_iam_user_details(token: str) -> dict[str, Any]:
    """Get IAM user details from token.

    :param token: IAM token
    :type token: str

    :returns: IAM user details
    :rtype: dict[str, Any]

    :raises: WMLClientError if there is an error getting IAM user details
    """
    try:
        token_info = _get_token_info(token)
    except json.JSONDecodeError as e:
        raise WMLClientError("Error getting IAM user details.") from e

    return token_info
