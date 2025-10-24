#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import asyncio
import base64
import json
import threading
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from ibm_watsonx_ai.utils.autoai.errors import TokenRemovedDuringClientCopy
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

STATUS_FORCELIST = (401, 500, 502, 503, 504, 520, 521, 524)


class BaseAuth(ABC):
    """Base class for any authentication method used in the APIClient."""

    _token: str | None = None

    def get_token(self) -> str:
        """Returns the token.

        :returns: token to be used with service
        :rtype: str
        """
        raise NotImplementedError()

    async def aget_token(self) -> str:
        """Returns the token.

        :returns: token to be used with service
        :rtype: str
        """
        raise NotImplementedError()


@dataclass
class TokenInfo:
    """Data class in which information about token and the token are returned to be put into auth method structures.
    `expiration_datetime` should be only set when token expiration information cannot be extracted from the token,
    and this is when token is not JWT token. Otherwise, `expiration_datetime` should be set to None.

    :param token: token to be used with service
    :type token: str

    :param expiration_datetime: datetime of token expiration, if the token is not JWT, otherwise should be set to None
    :type expiration_datetime: datetime or None, optional
    """

    token: str
    expiration_datetime: datetime | None = None


class RefreshableTokenAuth(BaseAuth, ABC):
    """Abstract base class of all authentication methods which are using token generation and refresh.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing

    :param refreshing_timedelta: time to expiration below which the token will be refreshed before use
    :type refreshing_timedelta: timedelta, optional
    """

    _hardcoded_expiration_datetime: datetime | None = None

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None,
        on_token_refresh: Callable[[], None] | None,
        refreshing_timedelta: timedelta | None = None,
    ) -> None:
        self._api_client = api_client
        self._on_token_creation = on_token_creation
        self._on_token_refresh = on_token_refresh
        self._refreshing_timedelta = refreshing_timedelta
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def get_token(self) -> str:
        """Returns the token. If the token will be about to expire, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        # serve token if it is ready and not refreshing without lock
        if self._token is not None and not self._is_refresh_needed():
            return self._token

        with self._lock:
            if self._token is None:
                self._save_token_data(self._generate_token())
                self._set_refreshing_timedelta_if_needed()

                if self._on_token_creation:
                    self._on_token_creation()
                return self._token

            if self._is_refresh_needed():
                self._save_token_data(self._refresh_token())
                if self._on_token_refresh:
                    self._on_token_refresh()

            return self._token

    async def aget_token(self) -> str:
        """Returns the token asynchronously. If the token will be about to expire, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        # serve token if it is ready and not refreshing without lock
        if self._token is not None and not self._is_refresh_needed():
            return self._token

        async with self._async_lock:
            if self._token is None:
                self._save_token_data(await self._agenerate_token())
                self._set_refreshing_timedelta_if_needed()

                if self._on_token_creation:
                    self._on_token_creation()
                return self._token

            if self._is_refresh_needed():
                self._save_token_data(await self._arefresh_token())
                if self._on_token_refresh:
                    self._on_token_refresh()

            return self._token

    def _set_refreshing_timedelta_if_needed(self):
        """Set refreshing timedelta basing on expiration time if no refreshing timedelta was passed in constructor."""
        time_to_expiration = self._get_expiration_datetime() - datetime.now()

        if self._refreshing_timedelta is None:
            if time_to_expiration > timedelta(minutes=30):
                self._refreshing_timedelta = timedelta(minutes=15)
            elif time_to_expiration > timedelta(minutes=3):
                # for minimal cloud token expiration = 15 min, the refreshing timedelta will be 5 min
                self._refreshing_timedelta = (time_to_expiration) / 3
            else:
                # for token expiration time < 3 min, the refreshing time is always 1 min,
                # which sometimes triggers refresh always (for expiration time < 1 min)
                self._refreshing_timedelta = timedelta(minutes=1)

    def _generate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        raise NotImplementedError()

    async def _agenerate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        raise NotImplementedError()

    def _refresh_token(
        self,
    ) -> TokenInfo:
        """Refresh token.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        # if not provided implementation, refresh is handled as generation from creds
        return self._generate_token()

    async def _arefresh_token(
        self,
    ) -> TokenInfo:
        """Refresh token.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        # if not provided implementation, refresh is handled as generation from creds
        return await self._agenerate_token()

    def _is_refresh_needed(self) -> bool:
        """Check if the time of expiration is below minimal expiration timedelta.

        :returns: result of check
        :rtype: bool
        """
        if exp_datetime := self._get_expiration_datetime():
            return exp_datetime - datetime.now() < self._refreshing_timedelta
        else:
            return True

    def _get_expiration_datetime(self) -> datetime:
        """Return expiration datetime. Implementation for JWT token.

        :returns: datetime of token expiration
        :rtype: datetime
        """
        if self._hardcoded_expiration_datetime is not None:
            return self._hardcoded_expiration_datetime

        token_info = _get_token_info(self._token)
        token_expire = token_info.get("exp")

        return datetime.fromtimestamp(token_expire)

    def _save_token_data(self, token_info: TokenInfo) -> None:
        """Write data from TokenInfo into authentication method fields for its mechanism to work properly.

        :param token_info: data of token returned after generation or refresh of token
        :type token_info: TokenInfo
        """
        self._token = token_info.token
        self._hardcoded_expiration_datetime = token_info.expiration_datetime


class TokenAuth(BaseAuth):
    """Basic authentication method, the object is keeping existing token and return it when asked.
    Token cannot be refreshed.

    :param token: token to be used with service
    :type token: str

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        token: str,
        on_token_set: Callable[[], None] | None = None,
    ) -> None:
        BaseAuth.__init__(self)
        self._on_token_set = on_token_set
        self.set_token(token)

    def get_token(self) -> str:
        """Returns the token. The token will not be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        return self._token

    async def aget_token(self) -> str:
        """Returns the token. The token will not be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        return self._token

    def set_token(self, token: str) -> None:
        """Set new token.

        :param token: token to be used with service
        :type token: str
        """
        WMLResource._validate_type(token, "token", str, mandatory=True)

        self._token = token
        if self._on_token_set:
            self._on_token_set()


class TokenRemovedDuringClientCopyPlaceholder(BaseAuth):
    """Placeholder which indicates that no auth is currently available until `APIClient.set_token(token)` is used."""

    def __init__(self) -> None:
        BaseAuth.__init__(self)

    def get_token(self) -> str:
        """Raise an error when `get_token()` is called."""
        raise TokenRemovedDuringClientCopy()


def get_auth_method(
    api_client: APIClient,
    on_token_set: Callable[[], None] | None = None,
    on_token_creation: Callable[[], None] | None = None,
    on_token_refresh: Callable[[], None] | None = None,
) -> BaseAuth:
    """
    Return authentication method using values from API client.

    :param api_client: APIClient object
    :type api_client: APIClient

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional

    :returns: authentication method object
    :rtype: BaseAuth
    """
    creds = api_client.credentials

    if (
        creds.token
        and not (creds._is_env_token and (creds.api_key or creds.password))
        and not api_client.credentials.trusted_profile_id
    ):
        # situation one of these:
        # - there is token passed by user (and may be password or apikey)
        # - there is token from env and no additional password or api_key in the credentials
        return TokenAuth(
            token=creds.token,
            on_token_set=on_token_set,
        )
    elif any(
        getattr(creds, key, None) for key in ["token_function", "atoken_function"]
    ):  # token function passed
        from ibm_watsonx_ai.utils.auth.jwt_token_function_auth import (
            JWTTokenFunctionAuth,
        )

        return JWTTokenFunctionAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )
    elif api_client.ICP_PLATFORM_SPACES:  # CPD
        from ibm_watsonx_ai.utils.auth.icp_auth import ICPAuth

        return ICPAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )
    elif (
        "aws" in api_client.credentials.url
        or "ibmforusgov" in api_client.credentials.url
    ):  # Cloud AWS or GovCloud
        from ibm_watsonx_ai.utils.auth.aws_auth import AWSTokenAuth

        return AWSTokenAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )
    elif api_client.credentials.trusted_profile_id:  # Cloud with trusted profile
        from ibm_watsonx_ai.utils.auth.trusted_profile_auth import TrustedProfileAuth

        return TrustedProfileAuth(
            api_client,
            token=creds.token,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
            on_token_set=on_token_set,
        )
    else:  # Cloud
        from ibm_watsonx_ai.utils.auth.iam_auth import IAMTokenAuth

        return IAMTokenAuth(
            api_client,
            on_token_creation=on_token_creation,
            on_token_refresh=on_token_refresh,
        )


def _get_token_info(token: str) -> dict[str, Any]:
    """Get info (aka payload part) from token.

    :param token: token with encoded information
    :type token: str

    :returns: info from token
    :rtype: dict[str, Any]

    """
    token_parts = token.split(".")
    token_padded = token_parts[1] + "==="

    try:
        token_info = json.loads(
            base64.b64decode(token_padded).decode("utf-8", errors="ignore")
        )
    except ValueError:
        # If there is a problem with decoding (e.g. special char in token), add altchars
        token_info = json.loads(
            base64.b64decode(token_padded, altchars="_-").decode(
                "utf-8", errors="ignore"
            )
        )

    return token_info
