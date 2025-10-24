#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import pandas as pd

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource


class Providers(WMLResource):
    """Model Gateway providers class."""

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

    def create(
        self, provider: str, name: str | None = None, data: dict | None = None
    ) -> dict:
        """Create provider in Model Gateway.

        :param provider: provider name
        :type provider: str

        :param name: name of provider for display
        :type name: str, optional

        :param data: data required to connect to provider api
        :type data: dict, optional

        :returns: provider details
        :rtype: dict
        """

        request_json = {}

        if name:
            request_json["name"] = name

        if data is not None:
            request_json["data"] = data

        response = self._client.httpx_client.post(
            self._client._href_definitions.get_gateway_provider_href(provider),
            headers=self._client._get_headers(),
            json=request_json,
        )

        return self._handle_response(201, "provider creation", response)

    def get_details(self, provider_id: str | None = None) -> dict:
        """Get provider/providers details:
         - `provider_id` is set - details for given provider are returned
         - `provider_id` is `None` - details for all providers are returned

        :param provider_id: unique provider ID
        :type provider_id: str, optional

        :returns: provider/providers details
        :rtype: dict
        """
        if provider_id:
            response = self._client.httpx_client.get(
                self._client._href_definitions.get_gateway_provider_href(provider_id),
                headers=self._client._get_headers(),
            )

            return self._handle_response(200, "getting provider details", response)
        else:
            response = self._client.httpx_client.get(
                self._client._href_definitions.get_gateway_providers_href(),
                headers=self._client._get_headers(),
            )

            return self._handle_response(200, "getting providers details", response)

    def get_available_models_details(self, provider_id: str) -> dict:
        """Get available models details for given provider.

        :param provider_id: unique provider ID
        :type provider_id: str

        :returns: details of available models for provider
        :rtype: dict
        """
        response = self._client.httpx_client.get(
            self._client._href_definitions.get_gateway_provider_available_models_href(
                provider_id
            ),
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            200, "getting provider available models details", response
        )

    def list(self) -> pd.DataFrame:
        """List providers.

        :returns: dataframe with providers details
        :rtype: pandas.DataFrame
        """
        providers_details = self.get_details()["data"]

        providers_values = [
            (
                m["uuid"],
                m["name"],
                m["type"],
            )
            for m in providers_details
        ]

        table = self._list(providers_values, ["ID", "NAME", "TYPE"], limit=None)

        return table

    def list_available_models(self, provider_id: str) -> pd.DataFrame:
        """List available models for provider.

        :param provider_id: unique provider ID
        :type provider_id: str

        :returns: dataframe with available models details
        :rtype: pandas.DataFrame
        """
        models_details = self.get_available_models_details(provider_id)["data"]

        if models_details is None:
            raise WMLClientError(
                f"Available models not supported for provider=`{provider_id}`."
            )

        models_values = [
            (
                m["id"],
                m["owned_by"],
            )
            for m in models_details
        ]

        table = self._list(models_values, ["MODEL_ID", "TYPE"], limit=None)

        return table

    def delete(self, provider_id: str) -> str:
        """Delete provider.

        :param provider_id: unique provider ID
        :type provider_id: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str
        """
        response = self._client.httpx_client.delete(
            self._client._href_definitions.get_gateway_provider_href(provider_id),
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204, "provider deletion", response, json_response=False
        )

    @staticmethod
    def get_id(provider_details: dict) -> str:
        """Get provider ID from provider details.

        :param provider_details: details of the provider in Model Gateway
        :type provider_details: dict

        :returns: unique provider ID
        :rtype: str
        """
        return provider_details["uuid"]
