#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ibm_watsonx_ai.utils.auth.base_auth import RefreshableTokenAuth, TokenInfo
from ibm_watsonx_ai.wml_client_error import WMLClientError

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class JWTTokenFunctionAuth(RefreshableTokenAuth):
    """Token function authentication method class.

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

        if all(
            not callable(getattr(self._api_client.credentials, attr_name, None))
            for attr_name in ("token_function", "atoken_function")
        ):
            raise WMLClientError(
                "Error getting token with token function: "
                "One of: 'token_function', 'atoken_function' is mandatory in credentials."
            )

    @staticmethod
    def _handle_token_function_result(token_function_result: Any):
        match token_function_result:
            case TokenInfo() as token_info:
                return token_info
            case str() as token:
                return TokenInfo(token)
            case _:
                raise WMLClientError(
                    "Value returned from `token_function` must be either "
                    "string containing token or `TokenInfo` object."
                )

    def _generate_token(self) -> TokenInfo:
        """Generate token using ``token_function`` provided in credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

        token_function = getattr(self._api_client.credentials, "token_function")

        return self._handle_token_function_result(
            token_function(self._api_client._session)
        )

    async def _agenerate_token(self) -> TokenInfo:
        """Generate token asynchronously using ``atoken_function`` provided in credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """

        atoken_function = getattr(self._api_client.credentials, "atoken_function")

        return self._handle_token_function_result(
            await atoken_function(self._api_client._session)
        )
