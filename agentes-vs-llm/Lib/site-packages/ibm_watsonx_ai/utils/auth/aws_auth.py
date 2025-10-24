#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.utils.auth.base_auth import (
    STATUS_FORCELIST,
    RefreshableTokenAuth,
    TokenInfo,
)
from ibm_watsonx_ai.utils.utils import _requests_retry_session
from ibm_watsonx_ai.wml_client_error import (
    AuthenticationError,
    InvalidCredentialsError,
    WMLClientError,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class AWSTokenAuth(RefreshableTokenAuth):
    """AWS IAM token authentication method class.

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
                "api_key for AWS IAM token is not provided in credentials for the client."
            )

    def _generate_token(self) -> TokenInfo:
        """Generate token using AWS IAM authentication.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        response = _requests_retry_session(status_forcelist=STATUS_FORCELIST).post(
            self._api_client._href_definitions.get_aws_token_url(),
            headers={"Content-Type": "application/json"},
            json={"apikey": self._api_client.credentials.api_key},
        )

        if response.status_code == 200:
            return TokenInfo(response.json().get("token"))
        elif 400 <= response.status_code < 500:
            raise InvalidCredentialsError(reason=response.text)
        else:
            raise AuthenticationError("AWS IAM", response)

    async def _agenerate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        response = await self._api_client.async_httpx_client.post(
            self._api_client._href_definitions.get_aws_token_url(),
            headers={"Content-Type": "application/json"},
            json={"apikey": self._api_client.credentials.api_key},
        )

        if response.status_code == 200:
            return TokenInfo(response.json().get("token"))
        elif 400 <= response.status_code < 500:
            raise InvalidCredentialsError(reason=response.text)
        else:
            raise AuthenticationError("AWS IAM", response)
