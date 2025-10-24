#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import datetime

import pandas as pd

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_resource import WMLResource


class Models(WMLResource):
    """Model Gateway models class."""

    def __init__(self, api_client: APIClient):
        WMLResource.__init__(self, __name__, api_client)

    def create(
        self,
        provider_id: str,
        model: str,
        alias: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Register model in Model Gateway.

        :param provider_id: unique provider ID obtained from provider details
        :type provider_id: str

        :param model: model name as supported by provider
        :type model: str

        :param alias: alias for registered model, can be used later as model name during
            embeddings or text/chat completions calls
        :type alias: str, optional

        :param metadata: additional metadata which can be added for the model
        :type metadata: dict, optional

        :returns: model details
        :rtype: dict
        """
        request_json = {"id": model}

        if alias:
            request_json["alias"] = alias

        if metadata is not None:
            request_json["metadata"] = metadata

        response = self._client.httpx_client.post(
            self._client._href_definitions.get_gateway_models_href(provider_id),
            headers=self._client._get_headers(),
            json=request_json,
        )

        return self._handle_response(201, "model creation", response)

    def get_details(
        self, *, model_id: str | None = None, provider_id: str | None = None
    ) -> dict:
        """Get details of model or models:
            - `model_id` is set - details for single model are returned, `provider_id` if set is ignored
            - `provider_id` is set, `model_id` is `None` - details for all models for given provider are returned
            - both `model_id` and `provider_id` are `None` - all models details are returned

        :param model_id: unique model ID
        :type model_id: str, optional

        :param provider_id: unique provider ID, ignored if `model_id` is set
        :type provider_id: str, optional

        :returns: details of model/models
        :rtype: dict
        """
        if model_id:
            response = self._client.httpx_client.get(
                self._client._href_definitions.get_gateway_model_href(model_id),
                headers=self._client._get_headers(),
            )

            return self._handle_response(200, "getting model details", response)
        elif provider_id:
            response = self._client.httpx_client.get(
                self._client._href_definitions.get_gateway_models_href(provider_id),
                headers=self._client._get_headers(),
            )

            return self._handle_response(200, "getting models details", response)
        else:
            response = self._client.httpx_client.get(
                self._client._href_definitions.get_gateway_all_tenant_models_href(),
                headers=self._client._get_headers(),
            )

            return self._handle_response(
                200, "getting all tenant models details", response
            )

    def list(self, provider_id: str | None = None) -> pd.DataFrame:
        """List models registered in Model Gateway. List can be filtered by `provider_id`.

        :param provider_id: ID of provider added into Model Gateway
        :type provider_id: str, optional

        :returns: dataframe containing list results
        :rtype: pandas.DataFrame
        """
        models_details = self.get_details(provider_id=provider_id)["data"]

        models_values = [
            (
                m["uuid"],
                m["id"],
                datetime.datetime.fromtimestamp(m["created"]),
                m["owned_by"],
            )
            for m in models_details
        ]

        table = self._list(
            models_values, ["ID", "MODEL", "CREATED", "TYPE"], limit=None
        )

        return table

    def delete(self, model_id: str) -> str:
        """Unregister model from Model Gateway.

        :param model_id: unique model ID obtained from model details
        :type model_id: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str
        """
        response = self._client.httpx_client.delete(
            self._client._href_definitions.get_gateway_model_href(model_id),
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204, "model deletion", response, json_response=False
        )

    @staticmethod
    def get_id(model_details: dict) -> str:
        """Get model ID from model details.

        :param model_details: details of the model registered in Model Gateway
        :type model_details: dict

        :returns: unique model ID
        :rtype: str
        """
        return model_details["uuid"]
