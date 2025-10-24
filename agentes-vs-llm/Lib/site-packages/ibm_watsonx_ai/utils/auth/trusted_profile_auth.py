#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.utils.auth import IAMTokenAuth
from ibm_watsonx_ai.utils.auth.base_auth import (
    STATUS_FORCELIST,
    RefreshableTokenAuth,
    TokenAuth,
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


class TrustedProfileAuth(RefreshableTokenAuth):
    """Trusted profile authentication method class.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param token: token to be used to generate trusted profile token
    :type token: str, optional

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        token: str | None = None,
        on_token_creation: Callable[[], None] | None = None,
        on_token_refresh: Callable[[], None] | None = None,
        on_token_set: Callable[[], None] | None = None,
    ) -> None:
        RefreshableTokenAuth.__init__(
            self, api_client, on_token_creation, on_token_refresh
        )
        self._trusted_profile_id = api_client.credentials.trusted_profile_id

        if token is None:

            def _on_token_refresh():
                self._save_token_data(self._generate_token())

            self._internal_auth_method = IAMTokenAuth(
                api_client,
                on_token_refresh=_on_token_refresh,
            )
        else:
            self._internal_auth_method = TokenAuth(token, on_token_set=on_token_set)

    def get_token(self) -> str:
        """Returns the trusted profile token. If `api_key` has been passed and the token will be about to expire, it will be refreshed.
        If classic token is set, a new trusted profile token will be generated and set.

        :returns: token to be used with service
        :rtype: str
        """
        if isinstance(self._internal_auth_method, IAMTokenAuth):
            self._internal_auth_method.get_token()  # trigger internal token refresh if needed, before refreshing profile token if needed
            return super().get_token()

        if not self._is_trusted_profile_token():
            # generate trusted profile token
            new_trusted_token = self._generate_token().token
            self.set_token(new_trusted_token)
            return new_trusted_token
        else:
            return self._internal_auth_method.get_token()

    def set_token(self, token: str) -> None:
        """Set new token.

        :param token: token to be used to generate trusted profile token
        :type token: str
        """
        if isinstance(self._internal_auth_method, TokenAuth):
            self._internal_auth_method.set_token(token)
        else:
            raise WMLClientError(
                "`set_token()` is supported only for `TrustedProfileAuth` initialized with token"
            )

    def _generate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = _requests_retry_session(status_forcelist=STATUS_FORCELIST).post(
            self._api_client._href_definitions.get_iam_token_url(),
            params={
                "grant_type": "urn:ibm:params:oauth:grant-type:assume",
                "access_token": self._internal_auth_method.get_token(),
                "profile_id": self._trusted_profile_id,
            },
            headers=headers,
        )

        if response.status_code == 200:
            return TokenInfo(response.json().get("access_token"))
        elif 400 <= response.status_code < 500:
            raise InvalidCredentialsError(reason=response.text)
        else:
            raise AuthenticationError("trusted profile IAM", response)

    async def _agenerate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = await self._api_client.async_httpx_client.post(
            self._api_client._href_definitions.get_iam_token_url(),
            params={
                "grant_type": "urn:ibm:params:oauth:grant-type:assume",
                "access_token": await self._internal_auth_method.aget_token(),
                "profile_id": self._trusted_profile_id,
            },
            headers=headers,
        )

        if response.status_code == 200:
            return TokenInfo(response.json().get("access_token"))
        elif 400 <= response.status_code < 500:
            raise InvalidCredentialsError(reason=response.text)
        else:
            raise AuthenticationError("trusted profile IAM", response)

    def _is_trusted_profile_token(self) -> bool:
        """Valid only for `TokenAuth` as internal auth method.

        :returns: `True` if trusted profile token is set
        :rtype: bool
        """
        token = self._internal_auth_method.get_token()
        return (
            "Profile" == _get_token_info(token).get("sub_type", "")
            if token is not None
            else False
        )
