#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd

from ibm_watsonx_ai.foundation_models.schema import (
    BaseSchema,
    TSForecastParameters,
)
from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials


class TSModelInference(WMLResource):
    """
    Instantiate the time series model interface

    :param model_id: type of model to use
    :type model_id: str, optional

    :param params: parameters to use during request generation
    :type params: dict, TSForecastParameters, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param deployment_id: ID of tuned model's deployment
    :type deployment_id: str, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import TSModelInference

        forecasting_params = {
            "prediction_length": 10
        }

        ts_model = TSModelInference(
            model_id="<TIME SERIES MODEL>",
            params=forecasting_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id=project_id
        )

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import TSModelInference

        forecasting_params = {
            "prediction_length": 10
        }

        ts_model = TSModelInference(
            deployment_id="<ID of deployed model>",
            params=forecasting_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
        )

    """

    def __init__(
        self,
        model_id: str | None = None,
        params: dict | TSForecastParameters | None = None,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
        deployment_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.deployment_id = deployment_id

        if self.model_id and self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="Both arguments were provided.",
            )
        elif not self.model_id and not self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="None of the arguments were provided.",
            )

        self.params = deepcopy(params) if params is not None else None

        TSModelInference._validate_type(
            params, "params", [dict, TSForecastParameters], False, True
        )

        if credentials:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials, verify=verify)
        elif api_client:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if not self.deployment_id:
            if space_id:
                self._client.set.default_space(space_id)
            elif project_id:
                self._client.set.default_project(project_id)
            elif not api_client:
                raise InvalidMultipleArguments(
                    params_names_list=["space_id", "project_id"],
                    reason="None of the arguments were provided.",
                )

        WMLResource.__init__(self, __name__, self._client)

    def forecast(
        self,
        data: dict | pd.DataFrame,
        params: dict | TSForecastParameters | None = None,
        future_data: dict | pd.DataFrame | None = None,
    ) -> dict:
        """
        Generates a forecast based on the provided data and model parameters.

        :param data: A payload of data matching the schema provided. For more information about the data limitation see the product documentation https://cloud.ibm.com/apidocs/watsonx-ai.
        :type data: dict, pd.DataFrame, required

        :param params: Contains basic metadata about your time series data input. These metadata are used by the server to understand which field represents a time stamp or which are unique identifiers for separating time series from different input channels.
        :type params: dict, TSForecastParameters, optional

        :param future_data: Available only for deployment (BYOM). Exogenous or supporting features that extend into the forecasting horizon (e.g., a weather forecast or calendar of special promotions) which are known in advance. future_data would be in the same format as data except that all timestamps would be in the forecast horizon and it would not include previously specified target_columns.
        :type future_data: dict, pd.DataFrame, optional

        **Example:**

        .. code-block:: python

            # number of elements in the array for each field must be at least 512, 1024, or 1536 depending on the model; for example 512 for ibm/granite-ttm-512-96-r2
            data = {
                    "date": [
                        "2017-10-02T16:00:00",
                        "2017-10-02T17:00:00",
                        "2017-10-02T18:00:00"
                        ...
                    ],
                    "HUFL": [
                        1.1,
                        2.2,
                        3.3
                        ...
                    ]
                }

            future_data = {
                    "date": [
                        "2018-10-02T16:00:00",
                        "2018-10-02T17:00:00",
                        "2018-10-02T18:00:00"
                        ...
                    ],
                    "HUFL": [
                        10.10,
                        20.20,
                        30.30
                        ...
                    ]
                }

            params =  {
                "timestamp_column": "date",
                "target_columns": [
                    "HUFL",
                ],
                "prediction_length": 10
                "freq": "1h"
            },

            # The number of elements in the array for each field must be the prediction length of the model depending on the model; for example 96 for ibm/granite-ttm-512-96-r2,

            response = ts_model.forecast(data=data, params=params, future_data=future_data)

            # Print all response
            print(response)

        """
        if not self.deployment_id:
            self._client._check_if_either_is_set()

        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="list")

        self._validate_type(data, "data", dict, True)

        payload: dict = (
            {"model_id": self.model_id, "data": data}
            if self.model_id
            else {"data": data}
        )

        if params is not None:
            parameters = deepcopy(params)

        elif self.params is not None:
            parameters = self.params

        else:
            parameters = None

        if isinstance(parameters, BaseSchema):
            parameters = parameters.to_dict()

        self._validate_type(parameters, "params", dict, True)
        if parameters is not None:
            self._validate_type(
                parameters.get("timestamp_column"), "timestamp_column", str, True
            )

        if parameters is not None and "prediction_length" in parameters:
            payload["parameters"] = {
                "prediction_length": parameters.pop("prediction_length")
            }

        payload["schema"] = parameters

        if future_data is not None:
            if isinstance(future_data, pd.DataFrame):
                future_data = future_data.to_dict(orient="list")

            payload["future_data"] = future_data

        if not self.deployment_id:
            if self._client.default_project_id:
                payload["project_id"] = self._client.default_project_id
            elif self._client.default_space_id:
                payload["space_id"] = self._client.default_space_id

        url = (
            self._client._href_definitions.get_deployment_time_series_href(
                deployment_id=self.deployment_id
            )
            if self.deployment_id
            else self._client._href_definitions.get_time_series_href()
        )

        response = self._client.httpx_client.post(
            url=url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "forecast", response)
