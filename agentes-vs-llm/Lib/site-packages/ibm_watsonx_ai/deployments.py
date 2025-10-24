#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Iterable,
    Literal,
    NoReturn,
    TypeAlias,
    cast,
)
from urllib.parse import parse_qs, urlparse
from warnings import warn

import httpx
import numpy as np

from ibm_watsonx_ai.href_definitions import is_id
from ibm_watsonx_ai.libs.repo.util.library_imports import LibraryChecker
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import (
    DecisionOptimizationMetaNames,
    DeploymentMetaNames,
    ScoringMetaNames,
)
from ibm_watsonx_ai.utils import (
    StatusLogger,
    get_from_json,
    print_text_header_h1,
    print_text_header_h2,
)
from ibm_watsonx_ai.utils.autoai.utils import all_logging_disabled
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    InvalidValue,
    MissingValue,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import pandas as pd

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models.inference import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
    from ibm_watsonx_ai.lifecycle import SpecStates

lib_checker = LibraryChecker()

ListType: TypeAlias = list
InferenceType: TypeAlias = Literal["text", "text_stream", "chat", "chat_stream"]


class Deployments(WMLResource):
    """Deploy and score published artifacts (models and functions)."""

    DEFAULT_CONCURRENCY_LIMIT = 8

    ConfigurationMetaNames = DeploymentMetaNames()
    ScoringMetaNames = ScoringMetaNames()
    DecisionOptimizationMetaNames = DecisionOptimizationMetaNames()

    class HardwareRequestSizes(str, Enum):
        """
        An enum class that represents the different hardware request sizes
        available.
        """

        Small = "gpu_s"
        Medium = "gpu_m"
        Large = "gpu_l"

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def _deployment_status_errors_handling(
        self, deployment_details: dict, operation_name: str, deployment_id: str
    ) -> NoReturn:
        try:
            if "failure" not in deployment_details["entity"]["status"]:
                print(deployment_details["entity"]["status"])
                raise WMLClientError(
                    f"Deployment {operation_name} failed for deployment id: {deployment_id}. "
                    f"Error: {deployment_details['entity']['status']['state']}"
                )

            errors = deployment_details["entity"]["status"]["failure"]["errors"]
            for error in errors:
                match error:
                    case str():
                        try:
                            error_obj = json.loads(error)
                            print(error_obj["message"])
                        except Exception:
                            print(error)
                    case dict():
                        print(error["message"])
                    case _:
                        print(error)

            raise WMLClientError(
                f"Deployment {operation_name} failed for deployment id: {deployment_id}. Errors: {errors}"
            )
        except WMLClientError:
            raise
        except Exception as e:
            self._logger.debug("Deployment %s failed", operation_name, exc_info=e)
            print(deployment_details["entity"]["status"]["failure"])
            raise WMLClientError(
                f"Deployment {operation_name} failed for deployment id: {deployment_id}."
            )

    def _prepare_create_meta_props(
        self, artifact_id: str, meta_props: dict[str, Any] | None, rev_id: str | None
    ) -> dict[str, Any]:
        if meta_props is None:
            raise WMLClientError("Invalid input. meta_props can not be empty.")

        if self._client.CLOUD_PLATFORM_SPACES and "r_shiny" in meta_props:
            raise WMLClientError("Shiny is not supported in this release")

        if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            base_model_id = meta_props.get(self.ConfigurationMetaNames.BASE_MODEL_ID)

            if isinstance(base_model_id, ModelTypes):
                meta_props[self.ConfigurationMetaNames.BASE_MODEL_ID] = (
                    base_model_id.value
                )

        result_meta_props = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props
        )

        if (
            "serving_name" in str(result_meta_props)
            and meta_props.get("serving_name", False)
            and "r_shiny" in str(result_meta_props)
        ):
            if "parameters" in result_meta_props["r_shiny"]:
                result_meta_props["r_shiny"]["parameters"]["serving_name"] = meta_props[
                    "serving_name"
                ]
            else:
                result_meta_props["r_shiny"]["parameters"] = {
                    "serving_name": meta_props["serving_name"]
                }

            if "online" in result_meta_props:
                del result_meta_props["online"]

        if "wml_instance_id" in meta_props:
            result_meta_props["wml_instance_id"] = meta_props["wml_instance_id"]

        result_meta_props["asset"] = result_meta_props.get("asset") or {
            "id": artifact_id
        }

        if rev_id is not None:
            result_meta_props["asset"]["rev"] = rev_id

        if self._client.default_project_id:
            result_meta_props["project_id"] = self._client.default_project_id
        else:
            result_meta_props["space_id"] = self._client.default_space_id

        # note: checking if artifact_id points to prompt_template
        if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
            with all_logging_disabled():
                try:
                    from ibm_watsonx_ai.foundation_models.prompts import (
                        PromptTemplateManager,
                    )

                    model_id = (
                        PromptTemplateManager(api_client=self._client)
                        .load_prompt(artifact_id)
                        .model_id
                    )
                except Exception:
                    pass  # Foundation models scenario should not impact other ML models' deployment scenario.
                else:
                    result_meta_props.pop("asset")
                    result_meta_props["prompt_template"] = {"id": artifact_id}
                    if (
                        DeploymentMetaNames.BASE_MODEL_ID not in result_meta_props
                        and DeploymentMetaNames.BASE_DEPLOYMENT_ID
                        not in result_meta_props
                    ):
                        result_meta_props.update(
                            {DeploymentMetaNames.BASE_MODEL_ID: model_id}
                        )
        # --- end note

        return result_meta_props

    def _process_create_response(
        self, response: httpx.Response, background_mode: Any
    ) -> dict:
        if response.status_code != 202:
            error_msg = "Deployment creation failed"
            reason = response.text
            print_text_header_h2(error_msg)
            print(reason)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

        deployment_details = response.json()

        if background_mode:
            background_mode_turned_on_warning = (
                "Background mode is turned on and deployment scoring will be available only when status of deployment will be `ready`. "
                "To check deployment status run `client.deployments.get_details(deployment_id)"
            )
            warn(background_mode_turned_on_warning)
            return deployment_details

        if (
            self._client.ICP_PLATFORM_SPACES
            and "online_url" in deployment_details["entity"]["status"]
        ):
            scoring_url = deployment_details["entity"]["status"]["online_url"]["url"]
            deployment_details["entity"]["status"]["online_url"]["url"] = (
                scoring_url.replace("https://ibm-nginx-svc:443", self._credentials.url)
            )

        return deployment_details

    def _wait_for_deployment_creation(self, artifact_id: str, deployment_details: dict):
        deployment_id = self.get_id(deployment_details)

        print_text_header_h1(
            f"Synchronous deployment creation for id: '{artifact_id}' started"
        )

        status = deployment_details["entity"]["status"]["state"]

        notifications = set()
        with StatusLogger(status) as status_logger:
            while True:
                time.sleep(5)
                deployment_details = self._client.deployments.get_details(
                    deployment_id, _silent=True
                )

                # this is wrong , needs to update for ICP
                if "system" in deployment_details:
                    notification = deployment_details["system"]["warnings"][0][
                        "message"
                    ]
                    if notification not in notifications:
                        print("\nNote: " + notification)
                        notifications.add(notification)

                status = deployment_details["entity"]["status"]["state"]
                status_logger.log_state(status)
                if status not in {"DEPLOY_IN_PROGRESS", "initializing"}:
                    break

        if status in {"DEPLOY_SUCCESS", "ready"}:
            print("")
            print_text_header_h2(
                f"Successfully finished deployment creation, deployment_id='{deployment_id}'"
            )
            return deployment_details

        print_text_header_h2("Deployment creation failed")
        self._deployment_status_errors_handling(
            deployment_details, "creation", deployment_id
        )

    def create(
        self,
        artifact_id: str | None = None,
        meta_props: dict | None = None,
        rev_id: str | None = None,
        **kwargs: dict,
    ) -> dict:
        """Create a deployment from an artifact. An artifact is a model or function that can be deployed.

        :param artifact_id: ID of the published artifact (the model or function ID)
        :type artifact_id: str

        :param meta_props: meta props. To see the available list of meta names, use:

            .. code-block:: python

                client.deployments.ConfigurationMetaNames.get()

        :type meta_props: dict, optional

        :param rev_id: revision ID of the deployment
        :type rev_id: str, optional

        :return: metadata of the created deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
                client.deployments.ConfigurationMetaNames.ONLINE: {},
                client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : {"id": "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"},
                client.deployments.ConfigurationMetaNames.SERVING_NAME : 'sample_deployment'
            }
            deployment_details = client.deployments.create(artifact_id, meta_props)

        """
        artifact_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=artifact_id, resource_name="artifact"
        )

        background_mode = kwargs.get("background_mode")

        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_deprecated = (
                "`rev_id` parameter type as int is deprecated, "
                "please convert to str instead"
            )
            warn(rev_id_as_int_deprecated, category=DeprecationWarning)
            rev_id = str(rev_id)

        Deployments._validate_type(artifact_id, "artifact_id", str, True)

        payload_meta_props = self._prepare_create_meta_props(
            artifact_id, meta_props, rev_id
        )

        response = self._client.httpx_client.post(
            self._client._href_definitions.get_deployments_href(),
            json=payload_meta_props,
            params=self._client._params(),  # version is mandatory
            headers=self._client._get_headers(),
        )

        deployment_details = self._process_create_response(response, background_mode)
        if background_mode:
            return deployment_details

        return self._wait_for_deployment_creation(artifact_id, deployment_details)

    async def _await_for_deployment_creation(
        self, artifact_id: str, deployment_details: dict
    ):
        deployment_id = self.get_id(deployment_details)

        print_text_header_h1(
            f"Synchronous deployment creation for id: '{artifact_id}' started"
        )

        status = deployment_details["entity"]["status"]["state"]

        notifications = set()
        with StatusLogger(status) as status_logger:
            while True:
                await asyncio.sleep(5)
                deployment_details = await self._client.deployments.aget_details(
                    deployment_id, _silent=True
                )

                # this is wrong , needs to update for ICP
                if "system" in deployment_details:
                    notification = deployment_details["system"]["warnings"][0][
                        "message"
                    ]
                    if notification not in notifications:
                        print("\nNote: " + notification)
                        notifications.add(notification)

                status = deployment_details["entity"]["status"]["state"]
                status_logger.log_state(status)
                if status not in {"DEPLOY_IN_PROGRESS", "initializing"}:
                    break

        if status in {"DEPLOY_SUCCESS", "ready"}:
            print("")
            print_text_header_h2(
                f"Successfully finished deployment creation, deployment_id='{deployment_id}'"
            )
            return deployment_details

        print_text_header_h2("Deployment creation failed")
        self._deployment_status_errors_handling(
            deployment_details, "creation", deployment_id
        )

    async def acreate(
        self,
        artifact_id: str,
        meta_props: dict | None = None,
        rev_id: str | None = None,
        **kwargs: dict,
    ) -> dict:
        """Create a deployment from an artifact asynchronously. An artifact is a model or function that can be deployed.

        :param artifact_id: ID of the published artifact (the model or function ID)
        :type artifact_id: str

        :param meta_props: meta props. To see the available list of meta names, use:

            .. code-block:: python

                client.deployments.ConfigurationMetaNames.get()

        :type meta_props: dict, optional

        :param rev_id: revision ID of the deployment
        :type rev_id: str, optional

        :return: metadata of the created deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
                client.deployments.ConfigurationMetaNames.ONLINE: {},
                client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : {"id": "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"},
                client.deployments.ConfigurationMetaNames.SERVING_NAME : 'sample_deployment'
            }
            deployment_details = await client.deployments.acreate(artifact_id, meta_props)

        """

        Deployments._validate_type(artifact_id, "artifact_id", str, True)

        payload_meta_props = self._prepare_create_meta_props(
            artifact_id, meta_props, rev_id
        )

        response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_deployments_href(),
            json=payload_meta_props,
            params=self._client._params(),  # version is mandatory
            headers=await self._client._aget_headers(),
        )

        background_mode = kwargs.get("background_mode")
        deployment_details = self._process_create_response(response, background_mode)
        if background_mode:
            return deployment_details

        return await self._await_for_deployment_creation(
            artifact_id, deployment_details
        )

    @staticmethod
    def get_uid(deployment_details: dict) -> str:
        """Get deployment_uid from the deployment details.

        *Deprecated:* Use ``get_id(deployment_details)`` instead.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment UID that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_uid = client.deployments.get_uid(deployment)

        """
        get_uid_deprecated_warning = (
            "`get_uid()` is deprecated and will be removed in future. "
            "Instead, please use `get_id()`."
        )
        warn(get_uid_deprecated_warning, category=DeprecationWarning)
        return Deployments.get_id(deployment_details)

    @staticmethod
    def get_id(deployment_details: dict) -> str:
        """Get the deployment ID from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment ID that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_id = client.deployments.get_id(deployment)

        """
        Deployments._validate_type(deployment_details, "deployment_details", dict, True)

        metadata = deployment_details["metadata"]
        deployment_id = metadata.get("id") or metadata.get("guid")
        if deployment_id is None:
            raise MissingValue("deployment_details.metadata.id")

        return deployment_id

    @staticmethod
    def get_href(deployment_details: dict) -> str:
        """Get deployment_href from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment href that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_href = client.deployments.get_href(deployment)

        """
        Deployments._validate_type(deployment_details, "deployment_details", dict, True)

        try:
            if "href" in deployment_details["metadata"]:
                url = get_from_json(deployment_details, ["metadata", "href"])
            else:
                url = "/ml/v4/deployments/{}".format(
                    deployment_details["metadata"]["id"]
                )
        except Exception as e:
            raise WMLClientError(
                "Getting deployment url from deployment details failed.", e
            )

        if url is None:
            raise MissingValue("deployment_details.metadata.href")

        return url

    def _get_serving_name_info(self, serving_name: str) -> tuple[int, Any | None]:
        """Get info about the serving name

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about the serving name: (<status_code>, <response json if any>)
        :rtype: tuple

        **Example:**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available("test")

        """
        params = {
            "serving_name": serving_name,
            "conflict": "true",
            "version": self._client.version_param,
        }

        res = self._client.httpx_client.get(
            self._client._href_definitions.get_deployments_href(),
            params=params,
            headers=self._client._get_headers(),
        )

        return res.status_code, (res.json() if res.status_code == 409 else None)

    def is_serving_name_available(self, serving_name: str) -> bool:
        """Check if the serving name is available for use.

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about whether the serving name is available
        :rtype: bool

        **Example:**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available("test")

        """
        status_code, _ = self._get_serving_name_info(serving_name)

        return status_code != 409

    async def _aget_serving_name_info(
        self, serving_name: str
    ) -> tuple[int, Any | None]:
        """Get info about the serving name

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about the serving name: (<status_code>, <response json if any>)
        :rtype: tuple

        **Example:**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available("test")

        """
        params = {
            "serving_name": serving_name,
            "conflict": "true",
            "version": self._client.version_param,
        }

        res = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_deployments_href(),
            params=params,
            headers=await self._client._aget_headers(),
        )

        if res.status_code == 409:
            response = res.json()
        else:
            response = None

        return res.status_code, response

    async def ais_serving_name_available(self, serving_name: str) -> bool:
        """Check if the serving name is available for use asynchronously.

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about whether the serving name is available
        :rtype: bool

        **Example:**

        .. code-block:: python

            is_available = await client.deployments.ais_serving_name_available("test")

        """
        status_code, _ = await self._aget_serving_name_info(serving_name)

        return status_code != 409

    def get_details(
        self,
        deployment_id: str | None = None,
        serving_name: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        _silent: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Get information about deployment(s).
        If deployment_id is not passed, all deployment details are returned.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str, optional

        :param serving_name: serving name that filters deployments
        :type serving_name: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `deployment_id` is None
        :type spec_state: SpecStates, optional

        :return: metadata of the deployment(s)
        :rtype: dict (if ``deployment_id`` is not None) or {"resources": [dict]} otherwise

        **Examples:**

        .. tab-set::

            .. tab-item:: Retrieve single deployment

                .. code-block:: python

                    deployment_details = client.deployments.get_details(deployment_id)
                    deployment_details = client.deployments.get_details(deployment_id=deployment_id)

            .. tab-item:: Retrieve multiple deployments

                .. code-block:: python

                    deployments_details = client.deployments.get_details()
                    deployments_details = client.deployments.get_details(limit=100)
                    deployments_details = client.deployments.get_details(limit=100, get_all=True)

            .. tab-item:: Retrieval using Generator

                .. code-block:: python

                    deployments_details = []
                    for entry in client.deployments.get_details(limit=100, asynchronous=True, get_all=True):
                        deployments_details.extend(entry["resources"])

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs,
            resource_id=deployment_id,
            resource_name="deployment",
            can_be_none=True,
        )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, False)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(f"'deployment_id' is not an id: '{deployment_id}'")

        url = self._client._href_definitions.get_deployments_href()

        query_params = self._client._params()

        if serving_name:
            query_params["serving_name"] = serving_name

        if deployment_id is None:
            filter_func = (
                self._get_filter_func_by_spec_state(spec_state) if spec_state else None
            )

            deployment_details = self._get_artifact_details(
                base_url=url,
                id=deployment_id,
                limit=limit,
                resource_name="deployments",
                query_params=query_params,
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )
        else:
            deployment_details = self._get_artifact_details(
                url,
                deployment_id,
                limit,
                "deployments",
                query_params=query_params,
            )

        if (
            not isinstance(deployment_details, Generator)
            and "system" in deployment_details
            and not _silent
        ):
            print("Note: " + deployment_details["system"]["warnings"][0]["message"])

        return deployment_details

    async def aget_details(
        self,
        deployment_id: str | None = None,
        serving_name: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        _silent: bool = False,
    ) -> dict:
        """Get information about deployment(s) asynchronously.
        If deployment_id is not passed, all deployment details are returned.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str, optional

        :param serving_name: serving name that filters deployments
        :type serving_name: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `deployment_id` is None
        :type spec_state: SpecStates, optional

        :return: metadata of the deployment(s)
        :rtype: dict (if ``deployment_id`` is not None) or {"resources": [dict]} otherwise

        **Examples:**

        .. tab-set::

            .. tab-item:: Retrieve single deployment

                .. code-block:: python

                    deployment_details = await client.deployments.aget_details(deployment_id)
                    deployment_details = await client.deployments.aget_details(deployment_id=deployment_id)

            .. tab-item:: Retrieve multiple deployments

                .. code-block:: python

                    deployments_details = await client.deployments.aget_details()
                    deployments_details = await client.deployments.aget_details(limit=100)
                    deployments_details = await client.deployments.aget_details(limit=100, get_all=True)

            .. tab-item:: Retrieval using Generator

                .. code-block:: python

                    deployments_details = []
                    async for entry in await client.deployments.aget_details(
                        limit=100, asynchronous=True, get_all=True
                    ):
                        deployments_details.extend(entry["resources"])

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, False)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(f"'deployment_id' is not an id: '{deployment_id}'")

        url = self._client._href_definitions.get_deployments_href()

        query_params = self._client._params()

        if serving_name:
            query_params["serving_name"] = serving_name

        if deployment_id is None:
            filter_func = (
                self._get_filter_func_by_spec_state(spec_state) if spec_state else None
            )

            deployment_details = await self._aget_artifact_details(
                base_url=url,
                id=deployment_id,
                limit=limit,
                resource_name="deployments",
                query_params=query_params,
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )
        else:
            deployment_details = await self._aget_artifact_details(
                url,
                deployment_id,
                limit,
                "deployments",
                query_params=query_params,
            )

        if (
            not isinstance(deployment_details, AsyncGenerator)
            and "system" in deployment_details
            and not _silent
        ):
            print("Note: " + deployment_details["system"]["warnings"][0]["message"])

        return deployment_details

    @staticmethod
    def get_scoring_href(deployment_details: dict) -> str:
        """Get scoring URL from deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: scoring endpoint URL that is used to make scoring requests
        :rtype: str

        **Example:**

        .. code-block:: python

            scoring_href = client.deployments.get_scoring_href(deployment)

        """

        Deployments._validate_type(deployment_details, "deployment", dict, True)

        scoring_url = None
        try:
            if deployment_details["entity"]["status"].get("online_url") is None:
                raise MissingValue(
                    "Getting scoring url for deployment failed. This functionality is available only for sync deployments"
                )

            scoring_url = deployment_details["entity"]["status"]["online_url"]["url"]
        except Exception as e:
            raise WMLClientError(
                "Getting scoring url for deployment failed. This functionality is available only for sync deployments",
                e,
            )

        if scoring_url is None:
            raise MissingValue("scoring_url missing in online_predictions")

        return scoring_url

    @staticmethod
    def get_serving_href(deployment_details: dict) -> str:
        """Get serving URL from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: serving endpoint URL that is used to make scoring requests
        :rtype: str

        **Example:**

        .. code-block:: python

            scoring_href = client.deployments.get_serving_href(deployment)

        """

        Deployments._validate_type(deployment_details, "deployment", dict, True)

        try:
            serving_name = get_from_json(
                deployment_details, ["entity", "online", "parameters", "serving_name"]
            )

            serving_url = next(
                url
                for url in deployment_details["entity"]["status"]["serving_urls"]
                if serving_name == url.split("/")[-2]
            )

            if serving_url:
                return serving_url

            raise MissingValue(
                "Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name."
            )
        except Exception as e:
            raise WMLClientError(
                "Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name.",
                e,
            )

    def delete(self, deployment_id: str | None = None, **kwargs: Any) -> str:
        """Delete a deployment.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client.deployments.delete(deployment_id)

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=deployment_id, resource_name="deployment"
        )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(f"'deployment_id' is not an id: '{deployment_id}'")

        response_delete = self._client.httpx_client.delete(
            self._client._href_definitions.get_deployment_href(deployment_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "deployment deletion", response_delete, False)

    async def adelete(self, deployment_id: str) -> str:
        """Delete a deployment asynchronously.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client.deployments.adelete(deployment_id)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(f"'deployment_id' is not an id: '{deployment_id}'")

        response_delete = await self._client.async_httpx_client.delete(
            self._client._href_definitions.get_deployment_href(deployment_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(204, "deployment deletion", response_delete, False)

    def _convert_scoring_values(self, scoring_values: Any) -> dict[str, list]:
        lib_checker.check_lib(lib_name="pandas")
        import pandas as pd  # pylint: disable=import-outside-toplevel

        result: dict[str, list] = {}

        match scoring_values:
            case pd.DataFrame():
                # replace nan with None
                scoring_values = scoring_values.where(pd.notnull(scoring_values), None)
                fields_names = scoring_values.columns.values.tolist()
                values = scoring_values.values.tolist()

                try:
                    # note: below code fails when there aren't any null values in a dataframe
                    values[pd.isnull(values)] = None
                except TypeError:
                    pass

                result["values"] = values
                if fields_names is not None:
                    result["fields"] = fields_names
            case np.ndarray():
                result["values"] = np.where(
                    pd.notnull(scoring_values),
                    scoring_values,
                    np.full(scoring_values.shape, None),
                ).tolist()
            case _:
                result["values"] = scoring_values

        return result

    def _prepare_scoring_payload(self, meta_props: dict[str, Any]) -> dict[str, Any]:
        payload = {"input_data": []}

        for scoring_request in meta_props[self.ScoringMetaNames.INPUT_DATA]:
            scoring_request.update(
                self._convert_scoring_values(scoring_request["values"])
            )

            payload["input_data"].append(scoring_request)

        if scoring_parameters := meta_props.get(
            self.ScoringMetaNames.SCORING_PARAMETERS
        ):
            payload["scoring_parameters"] = scoring_parameters

        return payload

    def score(
        self, deployment_id: str, meta_props: dict, transaction_id: str | None = None
    ) -> dict:
        """Make scoring requests against the deployed artifact.

        :param deployment_id: unique ID of the deployment to be scored
        :type deployment_id: str

        :param meta_props: meta props for scoring, use ``client.deployments.ScoringMetaNames.show()`` to view the list of ScoringMetaNames
        :type meta_props: dict

        :param transaction_id: transaction ID to be passed with the records during payload logging
        :type transaction_id: str, optional

        :return: scoring result that contains prediction and probability
        :rtype: dict

        .. note::

                * *client.deployments.ScoringMetaNames.INPUT_DATA* is the only metaname valid for sync scoring.
                * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA:
                [{'fields':
                    ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                    'values': [
                        ['M',23,'Single','Student'],
                        ['M',55,'Single','Executive']
                    ]
                }]
            }
            predictions = client.deployments.score(deployment_id, scoring_payload)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if meta_props.get(self.ScoringMetaNames.INPUT_DATA) is None:
            raise WMLClientError(
                "Scoring data input 'ScoringMetaNames.INPUT_DATA' is mandatory for scoring"
            )

        payload = self._prepare_scoring_payload(meta_props)

        headers = self._client._get_headers()
        if transaction_id is not None:
            headers["x-global-transaction-id"] = transaction_id

        params = self._client._params()
        del params["space_id"]

        response_scoring = self._client.httpx_client.post(
            self._client._href_definitions.get_deployment_predictions_href(
                deployment_id
            ),
            json=payload,
            params=params,  # version parameter is mandatory
            headers=headers,
        )

        return self._handle_response(200, "scoring", response_scoring)

    async def ascore(
        self, deployment_id: str, meta_props: dict, transaction_id: str | None = None
    ) -> dict:
        """Make scoring requests against the deployed artifact asynchronously.

        :param deployment_id: unique ID of the deployment to be scored
        :type deployment_id: str

        :param meta_props: meta props for scoring, use ``client.deployments.ScoringMetaNames.show()`` to view the list of ScoringMetaNames
        :type meta_props: dict

        :param transaction_id: transaction ID to be passed with the records during payload logging
        :type transaction_id: str, optional

        :return: scoring result that contains prediction and probability
        :rtype: dict

        .. note::

                * *client.deployments.ScoringMetaNames.INPUT_DATA* is the only metaname valid for sync scoring.
                * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA:
                [{'fields':
                    ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                    'values': [
                        ['M',23,'Single','Student'],
                        ['M',55,'Single','Executive']
                    ]
                }]
            }
            predictions = await client.deployments.ascore(deployment_id, scoring_payload)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if meta_props.get(self.ScoringMetaNames.INPUT_DATA) is None:
            raise WMLClientError(
                "Scoring data input 'ScoringMetaNames.INPUT_DATA' is mandatory for scoring"
            )

        payload = self._prepare_scoring_payload(meta_props)

        headers = await self._client._aget_headers()
        if transaction_id is not None:
            headers["x-global-transaction-id"] = transaction_id

        params = self._client._params()
        del params["space_id"]

        response_scoring = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_deployment_predictions_href(
                deployment_id
            ),
            json=payload,
            params=params,  # version parameter is mandatory
            headers=headers,
        )

        return self._handle_response(200, "scoring", response_scoring)

    def get_download_url(self, deployment_details: dict) -> str:
        """Get deployment download URL from the deployment details.

        **Warning:** This method is deprecated and will be removed in the future.

        :param deployment_details: created deployment details
        :type deployment_details: dict

        :return: deployment download URL that is used to get file deployment (for example: Core ML)
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_url = client.deployments.get_download_url(deployment)

        """

        raise WMLClientError("Downloading virtual deployment is no longer supported")

    @staticmethod
    def _enrich_asset_details_with_type(asset_details: dict, asset_type: str) -> dict:
        asset_details["metadata"]["asset_type"] = (
            "model"
            if asset_type == "prompt_tune" or asset_type.endswith("model")
            else asset_type
        )

        return asset_details

    def _build_table_row(
        self, assets_info: dict, resource: dict, include_software_spec_state: bool
    ) -> tuple:
        base_row = (
            resource["metadata"].get("guid", resource["metadata"]["id"]),
            resource["metadata"]["name"],
            resource["entity"]["status"]["state"],
            resource["metadata"]["created_at"],
            resource["entity"].get("deployed_asset_type", "unknown"),
        )

        if include_software_spec_state:
            prompt_template_id = get_from_json(
                resource, ["entity", "prompt_template", "id"]
            )
            asset_id = get_from_json(
                resource, ["entity", "asset", "id"], prompt_template_id
            )

            asset_details = assets_info.get(asset_id, {})
            base_row += (
                self._client.software_specifications._get_state(asset_details),
                self._client.software_specifications._get_replacement(asset_details),
            )

        return base_row

    def list(
        self,
        limit: int | None = None,
        artifact_type: str | None = None,
        include_software_spec_state: bool = True,
    ) -> pd.DataFrame:
        """Returns deployments in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param artifact_type: return only deployments with the specified artifact_type
        :type artifact_type: str, optional

        :param include_software_spec_state: include 'SPEC_STATE' and 'SPEC_REPLACEMENT' columns in deployments.
                                    This requires sending more requests and slows down execution, defaults to True
        :type include_software_spec_state: bool, optional

        :return: pandas.DataFrame with the listed deployments
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.deployments.list()

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        details = self.get_details(get_all=self._should_get_all_values(limit))
        resources = details["resources"]

        assets_info: dict[str, dict] = {}
        if include_software_spec_state:
            asset_types: set[str] = {
                r["entity"]["deployed_asset_type"] for r in resources
            }
            asset_specs: dict[str, Any] = {}

            if "prompt_tune" in asset_types or any(
                asset_type.endswith("model") for asset_type in asset_types
            ):
                asset_specs["model"] = self._client._models.get_details(get_all=True)
            if "function" in asset_types:
                asset_specs["function"] = self._client._functions.get_details(
                    get_all=True
                )
            if "py_script" in asset_types:
                asset_specs["py_script"] = self._client.script.get_details(get_all=True)
            if "ai_service" in asset_types:
                asset_specs["ai_service"] = self._client._ai_services.get_details(
                    get_all=True
                )

            assets_info = {
                resource["metadata"]["id"]: self._enrich_asset_details_with_type(
                    resource, asset_type
                )
                for asset_type, resources in asset_specs.items()
                for resource in resources["resources"]
            }

        values = []
        for i, resource in enumerate(resources):
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            # Ideally this needs to be on the server side
            if limit is not None and i >= limit:
                break

            resource_type = resource["entity"].get("deployed_asset_type", "unknown")
            if artifact_type is not None and resource_type != artifact_type:
                continue

            values.append(
                self._build_table_row(
                    assets_info, resource, include_software_spec_state
                )
            )

        headers = ["ID", "NAME", "STATE", "CREATED", "ARTIFACT_TYPE"]
        if include_software_spec_state:
            headers += ["SPEC_STATE", "SPEC_REPLACEMENT"]

        return self._list(values, headers, limit)

    def list_jobs(self, limit: int | None = None) -> "pd.DataFrame":
        """Return the async deployment jobs in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed deployment jobs
        :rtype: pandas.DataFrame

        .. note::

            This method list only async deployment jobs created for WML deployment.

        **Example:**

        .. code-block:: python

            client.deployments.list_jobs()

        """

        details = self.get_job_details(limit=limit)
        resources = details["resources"]
        values = []
        index = 0

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            if limit is not None and index == limit:
                break

            if "scoring" in m["entity"]:
                state = m["entity"]["scoring"]["status"]["state"]
            else:
                state = m["entity"]["decision_optimization"]["status"]["state"]

            deploy_id = m["entity"]["deployment"]["id"]
            values.append(
                (m["metadata"]["id"], state, m["metadata"]["created_at"], deploy_id)
            )

            index = index + 1

        table = self._list(
            values, ["JOB-ID", "STATE", "CREATED", "DEPLOYMENT-ID"], limit
        )

        return table

    def _get_deployable_asset_type(self, details: dict) -> str:
        url = details["entity"]["asset"]["id"]
        if "model" in url:
            return "model"
        elif "function" in url:
            return "function"
        else:
            return "unknown"

    def _is_patch_job(self, changes: dict) -> tuple[bool, str]:
        is_patch_job = changes.get("asset") is not None or any(
            key in changes
            for key in [
                self.ConfigurationMetaNames.PROMPT_TEMPLATE,
                self.ConfigurationMetaNames.SERVING_NAME,
                self.ConfigurationMetaNames.OWNER,
            ]
        )

        if not is_patch_job:
            return False, ""

        if changes.get("asset") is not None:
            patch_job_field = "ASSET"
        elif self.ConfigurationMetaNames.PROMPT_TEMPLATE in changes:
            patch_job_field = "PROMPT_TEMPLATE"
        elif self.ConfigurationMetaNames.SERVING_NAME in changes:
            patch_job_field = "SERVING_NAME"
        elif self.ConfigurationMetaNames.OWNER in changes:
            patch_job_field = "OWNER"
        else:
            raise WMLClientError("Unexpected patch job element.")

        if len(changes) > 1:
            msg = (
                f"When {patch_job_field} is being updated/patched, other fields cannot be updated. "
                f"If other fields are to be updated, try without {patch_job_field} update. "
                f"{patch_job_field} update triggers deployment with the new asset retaining "
                "the same deployment_id"
            )
            print(msg)
            raise WMLClientError(msg)

        return True, patch_job_field

    def _prepare_patch_payload(
        self, changes: dict, deployment_details: dict
    ) -> list[dict]:
        serving_name_change, new_serving_name = False, None
        if self.ConfigurationMetaNames.SERVING_NAME in changes:
            new_serving_name = changes.pop(self.ConfigurationMetaNames.SERVING_NAME)
            serving_name_change = True

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            deployment_details, changes, with_validation=True
        )

        if serving_name_change:
            replace = "serving_name" in get_from_json(
                deployment_details, ["entity", "online", "parameters"], []
            )
            patch_payload.append(
                {
                    "op": "replace" if replace else "add",
                    "path": "/online/parameters",
                    "value": {"serving_name": new_serving_name},
                }
            )

        return patch_payload

    def _handle_update_response(
        self,
        response: httpx.Response,
        is_patch_job: bool,
        patch_job_field: str,
        background_mode: bool,
    ) -> dict:
        if is_patch_job and response.status_code == 202:
            deployment_details = self._handle_response(
                202, "deployment asset patch", response
            )

            print(
                f"Since {patch_job_field} is patched, deployment need to be restarted."
            )
            if background_mode:
                print(
                    "Monitor the status using deployments.get_details(deployment_id) api"
                )

            return deployment_details

        if response.status_code == 202:
            return self._handle_response(202, "deployment scaling", response)

        return self._handle_response(200, "deployment patch", response)

    def _wait_for_deployment_update(self, deployment_id: str) -> dict:
        deployment_details = self.get_details(deployment_id, _silent=True)

        print_text_header_h1(f"Deployment update for id: '{deployment_id}' started")

        status = deployment_details["entity"]["status"]["state"]

        with StatusLogger(status) as status_logger:
            while True:
                time.sleep(5)
                deployment_details = self.get_details(deployment_id, _silent=True)
                status = deployment_details["entity"]["status"]["state"]
                status_logger.log_state(status)

                if status not in {"initializing", "updating"}:
                    break

        if (
            status == "ready"
            and "failure" not in deployment_details["entity"]["status"]
        ):
            # from apidocs: If any failures, deployment will be reverted back to the previous id/rev
            # and the failure message will be captured in 'failure' field in the response.
            print("")
            print_text_header_h2(
                f"Successfully finished deployment update, deployment_id='{deployment_id}'"
            )
            return deployment_details

        print_text_header_h2("Deployment update failed")
        self._deployment_status_errors_handling(
            deployment_details, "update", deployment_id
        )

    def update(
        self,
        deployment_id: str | None = None,
        changes: dict | None = None,
        background_mode: bool = False,
        **kwargs: Any,
    ) -> dict | None:
        """Updates existing deployment metadata. If ASSET is patched, then 'id' field is mandatory
        and it starts a deployment with the provided asset id/rev. Deployment ID remains the same.

        :param deployment_id: unique ID of deployment to be updated
        :type deployment_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated deployment
        :rtype: dict or None

        :param background_mode: indicator whether the update() method will run in the background (async) or not (sync), defaults to False
        :type background_mode: bool, optional

        **Examples**

        .. code-block:: python

            metadata = {client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment"}
            updated_deployment_details = client.deployments.update(deployment_id, changes=metadata)

            metadata = {client.deployments.ConfigurationMetaNames.ASSET: {  "id": "ca0cd864-4582-4732-b365-3165598dc945",
                                                                            "rev":"2" }}
            deployment_details = client.deployments.update(deployment_id, changes=metadata)

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=deployment_id, resource_name="deployment"
        )
        if changes is None:
            raise TypeError(
                "update() missing 1 required positional argument: 'changes'"
            )

        Deployments._validate_type(changes, "changes", dict, True)
        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        if ("asset" in changes and not changes["asset"]) and (
            "prompt_template" in changes and not changes["prompt_template"]
        ):
            msg = "ASSET/PROMPT_TEMPLATE cannot be empty. 'id' and 'rev' (only ASSET) fields are supported. 'id' is mandatory"
            print(msg)
            raise WMLClientError(msg)

        is_patch_job, patch_job_field = self._is_patch_job(changes)

        deployment_details = self.get_details(deployment_id, _silent=True)

        patch_payload = self._prepare_patch_payload(changes, deployment_details)

        response = self._client.httpx_client.patch(
            self._client._href_definitions.get_deployment_href(deployment_id),
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        deployment_details = self._handle_update_response(
            response, is_patch_job, patch_job_field, background_mode
        )
        if background_mode:
            return deployment_details

        if response.status_code not in {200, 202}:
            error_msg = "Deployment update failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

        return self._wait_for_deployment_update(deployment_id)

    async def _await_for_deployment_update(self, deployment_id: str):
        deployment_details = await self.aget_details(deployment_id, _silent=True)

        print_text_header_h1(f"Deployment update for id: '{deployment_id}' started")

        status = deployment_details["entity"]["status"]["state"]

        with StatusLogger(status) as status_logger:
            while True:
                await asyncio.sleep(5)
                deployment_details = await self.aget_details(
                    deployment_id, _silent=True
                )
                status = deployment_details["entity"]["status"]["state"]
                status_logger.log_state(status)

                if status not in {"initializing", "updating"}:
                    break

        if (
            status == "ready"
            and "failure" not in deployment_details["entity"]["status"]
        ):
            # from apidocs: If any failures, deployment will be reverted back to the previous id/rev
            # and the failure message will be captured in 'failure' field in the response.
            print("")
            print_text_header_h2(
                f"Successfully finished deployment update, deployment_id='{deployment_id}'"
            )
            return deployment_details

        print_text_header_h2("Deployment update failed")
        self._deployment_status_errors_handling(
            deployment_details, "update", deployment_id
        )

    async def aupdate(
        self,
        deployment_id: str,
        changes: dict,
        background_mode: bool = False,
    ) -> dict | None:
        """Updates existing deployment metadata asynchronously. If ASSET is patched, then 'id' field is mandatory
        and it starts a deployment with the provided asset id/rev. Deployment ID remains the same.

        :param deployment_id: unique ID of deployment to be updated
        :type deployment_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated deployment
        :rtype: dict or None

        :param background_mode: indicator whether the update() method will run in the background (async) or not (sync), defaults to False
        :type background_mode: bool, optional

        **Examples**

        .. code-block:: python

            metadata = {client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment"}
            updated_deployment_details = client.deployments.update(deployment_id, changes=metadata)

            metadata = {client.deployments.ConfigurationMetaNames.ASSET: {  "id": "ca0cd864-4582-4732-b365-3165598dc945",
                                                                            "rev":"2" }}
            deployment_details = await client.deployments.aupdate(deployment_id, changes=metadata)

        """

        Deployments._validate_type(changes, "changes", dict, True)
        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        if ("asset" in changes and not changes["asset"]) and (
            "prompt_template" in changes and not changes["prompt_template"]
        ):
            msg = "ASSET/PROMPT_TEMPLATE cannot be empty. 'id' and 'rev' (only ASSET) fields are supported. 'id' is mandatory"
            print(msg)
            raise WMLClientError(msg)

        is_patch_job, patch_job_field = self._is_patch_job(changes)

        deployment_details = await self.aget_details(deployment_id, _silent=True)

        patch_payload = self._prepare_patch_payload(changes, deployment_details)

        response = await self._client.async_httpx_client.patch(
            self._client._href_definitions.get_deployment_href(deployment_id),
            json=patch_payload,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        deployment_details = self._handle_update_response(
            response, is_patch_job, patch_job_field, background_mode
        )
        if background_mode:
            return deployment_details

        if response.status_code not in {200, 202}:
            error_msg = "Deployment update failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

        return await self._await_for_deployment_update(deployment_id)

    def _prepare_create_job_input_data_references(
        self, meta_props: dict[str, Any], payload: dict[str, Any]
    ):
        if "input_data_references" not in meta_props:
            return

        self._validate_type(
            meta_props.get("input_data_references"),
            "input_data_references",
            list,
            True,
        )

        modified_input_data_references = False
        input_data = cast(
            Iterable[Any], copy.deepcopy(meta_props["input_data_references"])
        )

        for input_data_fields in input_data:
            if "connection" not in input_data_fields:
                modified_input_data_references = True
                input_data_fields["connection"] = {}

        if not modified_input_data_references:
            return

        if "scoring" in payload:
            payload["scoring"]["input_data_references"] = input_data
        else:
            payload["decision_optimization"]["input_data_references"] = input_data

    def _prepare_create_job_output_data_reference(
        self, meta_props: dict[str, Any], payload: dict[str, Any]
    ):
        if "output_data_reference" not in meta_props:
            return

        Deployments._validate_type(
            meta_props.get("output_data_reference"),
            "output_data_reference",
            dict,
            True,
        )

        output_data = cast(dict, copy.deepcopy(meta_props["output_data_reference"]))
        if "connection" not in output_data:
            output_data["connection"] = {}
            payload["scoring"]["output_data_reference"] = output_data

    def _prepare_create_job_output_data_references(
        self, meta_props: dict[str, Any], payload: dict[str, Any]
    ):
        if "output_data_references" not in meta_props:
            return

        Deployments._validate_type(
            meta_props.get("output_data_references"),
            "output_data_references",
            list,
            True,
        )

        modified_output_data_references = False
        output_data = cast(
            Iterable[Any], copy.deepcopy(meta_props["output_data_references"])
        )

        for output_data_fields in output_data:
            if "connection" not in output_data_fields:
                modified_output_data_references = True
                output_data_fields["connection"] = {}

        if modified_output_data_references and "decision_optimization" in payload:
            payload["decision_optimization"]["output_data_references"] = output_data

    def _prepare_create_job_payload(
        self,
        deployment_id: str,
        asset_details: dict[str, Any],
        meta_props: dict[str, Any],
    ) -> dict[str, Any]:
        is_decision_optimization_job = False
        if (
            "wml_model" in asset_details["entity"]
            and "type" in asset_details["entity"]["wml_model"]
            and "do" in asset_details["entity"]["wml_model"]["type"]
        ):
            is_decision_optimization_job = True

        if is_decision_optimization_job:
            payload = self.DecisionOptimizationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )
        else:
            payload = self.ScoringMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

        scoring_data = None
        if "scoring" in payload and "input_data" in payload["scoring"]:
            scoring_data = payload["scoring"]["input_data"]

        if (
            "decision_optimization" in payload
            and "input_data" in payload["decision_optimization"]
        ):
            scoring_data = payload["decision_optimization"]["input_data"]

        if scoring_data is not None:
            score_payload: list[dict] = []
            for scoring_request in scoring_data:
                if "values" in scoring_request:
                    scoring_request.update(
                        self._convert_scoring_values(scoring_request["values"])
                    )
                score_payload.append(scoring_request)

            if is_decision_optimization_job:
                payload["decision_optimization"]["input_data"] = score_payload
            else:
                payload["scoring"]["input_data"] = score_payload

        self._prepare_create_job_input_data_references(meta_props, payload)
        self._prepare_create_job_output_data_reference(meta_props, payload)
        self._prepare_create_job_output_data_references(meta_props, payload)

        payload["deployment"] = {"id": deployment_id}
        payload["space_id"] = self._client.default_space_id

        if "hardware_spec" in meta_props:
            payload["hardware_spec"] = meta_props[
                self.ConfigurationMetaNames.HARDWARE_SPEC
            ]

        if "hybrid_pipeline_hardware_specs" in meta_props:
            payload["hybrid_pipeline_hardware_specs"] = meta_props[
                self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS
            ]

        if "name" not in payload:
            payload["name"] = f"name_{uuid.uuid4()}"

        return payload

    def _score_async(
        self,
        deployment_id: str,
        scoring_payload: dict,
        transaction_id: str | None = None,
        retention: int | None = None,
    ) -> str | dict:
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(scoring_payload, "scoring_payload", dict, True)

        headers = self._client._get_headers()
        if transaction_id is not None:
            headers["x-global-transaction-id"] = transaction_id

        params = self._client._params()
        if not self._client.ICP_PLATFORM_SPACES and retention is not None:
            if not isinstance(retention, int) or retention < -1:
                raise TypeError(
                    "`retention` takes integer values greater or equal than -1."
                )

            params["retention"] = retention

        response_scoring = self._client.httpx_client.post(
            self._client._href_definitions.get_async_deployment_job_href(),
            params=params,
            json=scoring_payload,
            headers=headers,
        )

        return self._handle_response(202, "scoring asynchronously", response_scoring)

    def create_job(
        self,
        deployment_id: str,
        meta_props: dict,
        retention: int | None = None,
        transaction_id: str | None = None,
        _asset_id: str | None = None,
    ) -> str | dict:
        """Create an asynchronous deployment job.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param meta_props: meta props. To see the available list of metanames,
            use ``client.deployments.ScoringMetaNames.get()``
            or ``client.deployments.DecisionOptimizationMetaNames.get()``

        :type meta_props: dict

        :param retention: how many job days job meta should be retained,
            takes integer values >= -1, supported only on Cloud
        :type retention: int, optional

        :param transaction_id: transaction ID to be passed with the payload
        :type transaction_id: str, optional

        :return: metadata of the created async deployment job
        :rtype: dict or str

        .. note::

            * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {
                client.deployments.ScoringMetaNames.INPUT_DATA: [
                    {
                        'fields': ['GENDER', 'AGE', 'MARITAL_STATUS', 'PROFESSION'],
                        'values': [
                            ['M', 23, 'Single', 'Student'],
                            ['M', 55, 'Single', 'Executive'],
                        ],
                    }
                ]
            }
            async_job = client.deployments.create_job(deployment_id, scoring_payload)

        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if _asset_id:
            Deployments._validate_type(_asset_id, "_asset_id", str, True)
            # We assume that _asset_id is the id of the asset that was deployed
            # in the deployment with id deployment_id, and we save one REST call
            asset = _asset_id
        else:
            deployment_details = self.get_details(deployment_id)
            asset = deployment_details["entity"]["asset"]["id"]

        asset_details = self._client.data_assets.get_details(asset)
        payload = self._prepare_create_job_payload(
            deployment_id, asset_details, meta_props
        )

        return self._score_async(
            deployment_id, payload, transaction_id=transaction_id, retention=retention
        )

    async def _ascore_async(
        self,
        deployment_id: str,
        scoring_payload: dict,
        transaction_id: str | None = None,
        retention: int | None = None,
    ) -> str | dict:
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(scoring_payload, "scoring_payload", dict, True)

        headers = await self._client._aget_headers()
        if transaction_id is not None:
            headers["x-global-transaction-id"] = transaction_id

        params = self._client._params()
        if not self._client.ICP_PLATFORM_SPACES and retention is not None:
            if not isinstance(retention, int) or retention < -1:
                raise TypeError(
                    "`retention` takes integer values greater or equal than -1."
                )

            params["retention"] = retention

        response_scoring = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_async_deployment_job_href(),
            params=params,
            json=scoring_payload,
            headers=headers,
        )

        return self._handle_response(202, "scoring asynchronously", response_scoring)

    async def acreate_job(
        self,
        deployment_id: str,
        meta_props: dict,
        retention: int | None = None,
        transaction_id: str | None = None,
        _asset_id: str | None = None,
    ) -> str | dict:
        """Create an asynchronous deployment job asynchronously.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param meta_props: meta props. To see the available list of metanames,
            use ``client.deployments.ScoringMetaNames.get()``
            or ``client.deployments.DecisionOptimizationMetaNames.get()``

        :type meta_props: dict

        :param retention: how many job days job meta should be retained,
            takes integer values >= -1, supported only on Cloud
        :type retention: int, optional

        :param transaction_id: transaction ID to be passed with the payload
        :type transaction_id: str, optional

        :return: metadata of the created async deployment job
        :rtype: dict or str

        .. note::

            * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {
                client.deployments.ScoringMetaNames.INPUT_DATA: [
                    {
                        'fields': ['GENDER', 'AGE', 'MARITAL_STATUS', 'PROFESSION'],
                        'values': [
                            ['M', 23, 'Single', 'Student'],
                            ['M', 55, 'Single', 'Executive'],
                        ],
                    }
                ]
            }
            async_job = await client.deployments.acreate_job(deployment_id, scoring_payload)

        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if _asset_id:
            Deployments._validate_type(_asset_id, "_asset_id", str, True)
            # We assume that _asset_id is the id of the asset that was deployed
            # in the deployment with id deployment_id, and we save one REST call
            asset = _asset_id
        else:
            deployment_details = await self.aget_details(deployment_id)
            asset = deployment_details["entity"]["asset"]["id"]

        asset_details = await self._client.data_assets.aget_details(asset)
        payload = self._prepare_create_job_payload(
            deployment_id, asset_details, meta_props
        )

        return await self._ascore_async(
            deployment_id, payload, transaction_id=transaction_id, retention=retention
        )

    def get_job_details(
        self,
        job_id: str | None = None,
        include: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get information about deployment job(s).
        If deployment job_id is not passed, all deployment jobs details are returned.

        :param job_id: unique ID of the job
        :type job_id: str, optional

        :param include: fields to be retrieved from 'decision_optimization'
            and 'scoring' section mentioned as value(s) (comma separated) as output response fields
        :type include: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of deployment job(s)
        :rtype: dict (if job_id is not None) or {"resources": [dict]} (if job_id is None)

        **Example:**

        .. code-block:: python

            deployment_details = client.deployments.get_job_details()
            deployments_details = client.deployments.get_job_details(job_id=job_id)

        """
        job_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=job_id, resource_name="job", can_be_none=True
        )

        Deployments._validate_type(job_id, "job_id", str, False)

        params = self._client._params()
        if include:
            params["include"] = include

        return self._get_artifact_details(
            base_url=self._client._href_definitions.get_async_deployment_job_href(),
            id=job_id,
            limit=limit,
            resource_name="async deployment job" if job_id else "async deployment jobs",
            query_params=params,
        )

    async def aget_job_details(
        self,
        job_id: str | None = None,
        include: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Get information about deployment job(s) asynchronously.
        If deployment job_id is not passed, all deployment jobs details are returned.

        :param job_id: unique ID of the job
        :type job_id: str, optional

        :param include: fields to be retrieved from 'decision_optimization'
            and 'scoring' section mentioned as value(s) (comma separated) as output response fields
        :type include: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of deployment job(s)
        :rtype: dict (if job_id is not None) or {"resources": [dict]} (if job_id is None)

        **Example:**

        .. code-block:: python

            deployment_details = await client.deployments.aget_job_details()
            deployments_details = await client.deployments.aget_job_details(job_id=job_id)

        """

        Deployments._validate_type(job_id, "job_id", str, False)

        params = self._client._params()
        if include:
            params["include"] = include

        return await self._aget_artifact_details(
            base_url=self._client._href_definitions.get_async_deployment_job_href(),
            id=job_id,
            limit=limit,
            resource_name="async deployment job" if job_id else "async deployment jobs",
            query_params=params,
        )

    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a deployment job.

        :param job_id: unique ID of the deployment job
        :type job_id: str

        :return: status of the deployment job
        :rtype: dict

        **Example:**

        .. code-block:: python

            job_status = client.deployments.get_job_status(job_id)

        """

        job_details = self.get_job_details(job_id)

        if "scoring" not in job_details["entity"]:
            return job_details["entity"]["decision_optimization"]["status"]

        return job_details["entity"]["scoring"]["status"]

    async def aget_job_status(self, job_id: str) -> dict:
        """Get the status of a deployment job asynchronously.

        :param job_id: unique ID of the deployment job
        :type job_id: str

        :return: status of the deployment job
        :rtype: dict

        **Example:**

        .. code-block:: python

            job_status = await client.deployments.aget_job_status(job_id)

        """

        job_details = await self.aget_job_details(job_id)

        if "scoring" not in job_details["entity"]:
            return job_details["entity"]["decision_optimization"]["status"]

        return job_details["entity"]["scoring"]["status"]

    def get_job_id(self, job_details: dict) -> str:
        """Get the unique ID of a deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: unique ID of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_id=job_id)
            job_status = client.deployments.get_job_id(job_details)

        """
        return job_details["metadata"]["id"]

    def get_job_uid(self, job_details: dict) -> str:
        """Get the unique ID of a deployment job.

        *Deprecated:* Use ``get_job_id(job_details)`` instead.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: unique ID of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_uid=job_uid)
            job_status = client.deployments.get_job_uid(job_details)

        """
        get_job_uid_deprecated_warning = (
            "`get_job_uid()` is deprecated and will be removed in future. "
            "Instead, please use `get_job_id()`."
        )
        warn(get_job_uid_deprecated_warning, category=DeprecationWarning)
        return self.get_job_id(job_details)

    def get_job_href(self, job_details: dict) -> str:
        """Get the href of a deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: href of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_id=job_id)
            job_status = client.deployments.get_job_href(job_details)

        """
        return f"/ml/v4/deployment_jobs/{job_details['metadata']['id']}"

    def delete_job(
        self, job_id: str | None = None, hard_delete: bool = False, **kwargs: Any
    ) -> str:
        """Delete a deployment job that is running. This method can also delete metadata
        details of completed or canceled jobs when hard_delete parameter is set to True.

        :param job_id: unique ID of the deployment job to be deleted
        :type job_id: str

        :param hard_delete: specify `True` or `False`:

            `True` - To delete the completed or canceled job.

            `False` - To cancel the currently running deployment job.

        :type hard_delete: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client.deployments.delete_job(job_id)

        """
        job_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=job_id, resource_name="job"
        )
        Deployments._validate_type(job_id, "job_id", str, True)

        if job_id is not None and not is_id(job_id):
            raise WMLClientError(f"'job_id' is not an id: '{job_id}'")

        params = self._client._params()

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version <= 5.1:
            # for CPD 5.1 and lower there is need to use the jobs api directly.
            # From CPD 5.2.x + and Cloud deployment service will cover the call in DELETE /ml/v4/deployment_jobs
            # issue: #48242
            try:
                job_details = self.get_job_details(job_id=job_id)
                run_id = job_details["entity"]["platform_job"]["run_id"]

                response_delete = self._client.httpx_client.delete(
                    self._client._href_definitions.get_jobs_runs_href(
                        job_id=job_id, run_id=run_id
                    ),
                    params=params,
                    headers=self._client._get_headers(),
                )

                return self._handle_response(
                    204, "deployment async job deletion", response_delete, False
                )
            except Exception:
                pass

        if hard_delete is True:
            params["hard_delete"] = "true"

        response_delete = self._client.httpx_client.delete(
            self._client._href_definitions.get_async_deployment_jobs_href(job_id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204, "deployment async job deletion", response_delete, False
        )

    async def adelete_job(self, job_id: str, hard_delete: bool = False) -> str:
        """Delete a deployment job that is running asynchronously. This method can also delete metadata
        details of completed or canceled jobs when hard_delete parameter is set to True.

        :param job_id: unique ID of the deployment job to be deleted
        :type job_id: str

        :param hard_delete: specify `True` or `False`:

            `True` - To delete the completed or canceled job.

            `False` - To cancel the currently running deployment job.

        :type hard_delete: bool, optional


        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client.deployments.adelete_job(job_id)

        """
        Deployments._validate_type(job_id, "job_id", str, True)

        if job_id is not None and not is_id(job_id):
            raise WMLClientError(f"'job_id' is not an id: '{job_id}'")

        params = self._client._params()

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version <= 5.1:
            # for CPD 5.1 and lower there is need to use the jobs api directly.
            # From CPD 5.2.x + and Cloud deployment service will cover the call in DELETE /ml/v4/deployment_jobs
            # issue: #48242
            try:
                job_details = await self.aget_job_details(job_id=job_id)
                run_id = job_details["entity"]["platform_job"]["run_id"]

                response_delete = await self._client.async_httpx_client.delete(
                    self._client._href_definitions.get_jobs_runs_href(
                        job_id=job_id, run_id=run_id
                    ),
                    params=params,
                    headers=await self._client._aget_headers(),
                )

                return self._handle_response(
                    204, "deployment async job deletion", response_delete, False
                )
            except Exception:
                pass

        if hard_delete is True:
            params["hard_delete"] = "true"

        response_delete = await self._client.async_httpx_client.delete(
            self._client._href_definitions.get_async_deployment_jobs_href(job_id),
            params=params,
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(
            204, "deployment async job deletion", response_delete, False
        )

    def _get_filter_func_by_spec_state(
        self, spec_state: SpecStates
    ) -> Callable[[list], list[str]]:
        def filter_func(resources: list) -> list[str]:
            asset_ids = [
                i["metadata"]["id"]
                for key, value in {
                    "model": self._client._models.get_details(
                        get_all=True, spec_state=spec_state
                    ),
                    "function": self._client._functions.get_details(
                        get_all=True, spec_state=spec_state
                    ),
                }.items()
                for i in value["resources"]
            ]

            return [
                r
                for r in resources
                if get_from_json(r, ["entity", "asset", "id"]) in asset_ids
            ]

        return filter_func

    def _get_model_inference_url(
        self, deployment_id: str, inference_type: InferenceType
    ) -> str:
        match inference_type:
            case "text":
                return self._client._href_definitions.get_fm_deployment_generation_href(
                    deployment_id=deployment_id, item="text"
                )
            case "text_stream" if self._client._use_fm_ga_api:
                if self._client._use_fm_ga_api:
                    return self._client._href_definitions.get_fm_deployment_generation_stream_href(
                        deployment_id=deployment_id
                    )

                # Remove on CPD 5.0 release
                return self._client._href_definitions.get_fm_deployment_generation_href(
                    deployment_id=deployment_id, item="text_stream"
                )
            case "chat":
                return self._client._href_definitions.get_fm_deployment_chat_href(
                    deployment_id=deployment_id
                )
            case "chat_stream":
                return (
                    self._client._href_definitions.get_fm_deployment_chat_stream_href(
                        deployment_id=deployment_id
                    )
                )
            case _:
                raise InvalidValue(
                    value_name="inference_type",
                    reason=(
                        "Available types: 'text', 'text_stream', 'chat', 'chat_stream', "
                        f"got: {inference_type}."
                    ),
                )

    def _validate_get_model_inference_url(
        self,
        inference_type: InferenceType,
        deployment_details: dict[str, Any],
        generated_url: str,
    ):
        inference_url_list = [
            url.get("url")
            for url in get_from_json(
                deployment_details, ["entity", "status", "inference"], []
            )
        ]

        if not inference_url_list:
            inference_url_list = get_from_json(
                deployment_details, ["entity", "status", "serving_urls"], []
            )

        if (
            inference_type in ["text", "text_stream"]
            and generated_url not in inference_url_list
            and all(
                "/text/generation" not in inference_url
                for inference_url in inference_url_list
            )
        ):
            raise WMLClientError(
                Messages.get_message(
                    self.get_id(deployment_details),
                    message_id="fm_deployment_has_not_inference_for_generation",
                )
            )

    def _get_model_inference(
        self,
        deployment_id: str,
        inference_type: InferenceType,
        params: dict | None = None,
    ) -> "ModelInference":
        """
        Based on provided `deployment_id` and params get `ModelInference` instance.
        Verify that the deployment with the given `deployment_id` has generating methods.
        """
        # Import ModelInference here to avoid circular import error
        from ibm_watsonx_ai.foundation_models.inference import (
            ModelInference,  # pylint: disable=import-outside-toplevel
        )

        generated_url = self._get_model_inference_url(deployment_id, inference_type)

        deployment_details = self.get_details(deployment_id, _silent=True)

        self._validate_get_model_inference_url(
            inference_type, deployment_details, generated_url
        )

        return ModelInference(
            deployment_id=deployment_id, params=params, api_client=self._client
        )

    async def _aget_model_inference(
        self,
        deployment_id: str,
        inference_type: InferenceType,
        params: dict | None = None,
    ) -> "ModelInference":
        """
        Based on provided deployment_id and params get ModelInference instance.
        Verify that the deployment with the given deployment_id has generating methods.
        """
        # Import ModelInference here to avoid circular import error
        from ibm_watsonx_ai.foundation_models.inference import (
            ModelInference,  # pylint: disable=import-outside-toplevel
        )

        generated_url = self._get_model_inference_url(deployment_id, inference_type)

        deployment_details = await self.aget_details(deployment_id, _silent=True)

        self._validate_get_model_inference_url(
            inference_type, deployment_details, generated_url
        )

        return ModelInference(
            deployment_id=deployment_id, params=params, api_client=self._client
        )

    def generate(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> dict:
        """Generate a raw response with `prompt` for given `deployment_id`.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: prompt needed for text generation. If deployment_id points to the Prompt Template asset, then the prompt argument must be None, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param concurrency_limit: number of requests to be sent in parallel, maximum is 10
        :type concurrency_limit: int, optional

        :param async_mode: If True, then yield results asynchronously (using generator). In this case both the prompt and
                           the generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: scoring result containing generated content
        :rtype: dict
        """
        d_inference = self._get_model_inference(deployment_id, "text", params)
        return d_inference.generate(
            prompt=prompt,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            params=params,
            async_mode=async_mode,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    async def agenerate(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> dict:
        """Generate a raw response with `prompt` for given `deployment_id` asynchronously.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: prompt needed for text generation. If deployment_id points to the Prompt Template asset, then the prompt argument must be None, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: scoring result containing generated content
        :rtype: dict
        """
        d_inference = await self._aget_model_inference(deployment_id, "text", params)
        return await d_inference.agenerate(
            prompt=prompt,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            params=params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    async def agenerate_stream(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> AsyncGenerator:
        """Generate a raw response with `prompt` for given `deployment_id`.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: prompt needed for text generation. If deployment_id points to the Prompt Template asset, then the prompt argument must be None, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: scoring result containing generated content
        :rtype: dict
        """
        d_inference = await self._aget_model_inference(
            deployment_id, "text_stream", params
        )
        return await d_inference.agenerate_stream(
            prompt=prompt,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            params=params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    def generate_text(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> str:
        """Given the selected deployment (deployment_id), a text prompt as input, and the parameters and concurrency_limit,
        the selected inference will generate a completion text as generated_text response.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: the prompt string or list of strings. If the list of strings is passed, requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param raw_response: returns the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param concurrency_limit: number of requests to be sent in parallel, maximum is 10
        :type concurrency_limit: int, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """
        d_inference = self._get_model_inference(deployment_id, "text", params)
        return d_inference.generate_text(
            prompt=prompt,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            params=params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    def generate_text_stream(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
        guardrails_granite_guardian_params: dict | None = None,
    ) -> Generator:
        """Given the selected deployment (deployment_id), a text prompt as input and parameters,
        the selected inference will generate a streamed text as generate_text_stream.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: the prompt string, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :param guardrails_granite_guardian_params: parameters for Granite Guardian moderations
        :type guardrails_granite_guardian_params: dict, optional

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """
        d_inference = self._get_model_inference(deployment_id, "text_stream", params)
        return d_inference.generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
            guardrails_granite_guardian_params=guardrails_granite_guardian_params,
        )

    def chat(
        self,
        deployment_id: str,
        messages: ListType[dict],
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        params: dict | TextChatParameters | None = None,
    ) -> dict:
        d_inference = self._get_model_inference(deployment_id, "chat")
        return d_inference.chat(
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    def chat_stream(
        self,
        deployment_id: str,
        messages: ListType[dict],
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        params: dict | TextChatParameters | None = None,
    ) -> Generator:
        d_inference = self._get_model_inference(deployment_id, "chat_stream")
        return d_inference.chat_stream(
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    async def achat(
        self,
        deployment_id: str,
        messages: ListType[dict],
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        params: dict | TextChatParameters | None = None,
    ) -> dict:
        d_inference = await self._aget_model_inference(deployment_id, "chat")
        return await d_inference.achat(
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    async def achat_stream(
        self,
        deployment_id: str,
        messages: ListType[dict],
        context: str | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        params: dict | TextChatParameters | None = None,
    ) -> AsyncGenerator:
        d_inference = await self._aget_model_inference(deployment_id, "chat_stream")
        return await d_inference.achat_stream(
            messages=messages,
            params=params,
            context=context,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    def run_ai_service(
        self,
        deployment_id: str,
        ai_service_payload: dict,
        path_suffix: str | None = None,
    ) -> Any:
        """Execute an AI service by providing a scoring payload.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :param path_suffix: path suffix to be appended to the scoring url, defaults to None
        :type path_suffix: str, optional

        :return: response of the AI service
        :rtype: Any

        .. note::
            * By executing this class method, a POST request is performed.
            * In case of `method not allowed` error, try sending requests directly to your deployed ai service.
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        url = self._client._href_definitions.get_deployment_ai_service_href(
            deployment_id
        )
        if path_suffix is not None:
            url += f"/{path_suffix}"

        response_scoring = self._client.httpx_client.post(
            url=url,
            json=ai_service_payload,
            params=self._client._params(
                skip_for_create=True, skip_userfs=True
            ),  # version parameter is mandatory
            headers=self._client._get_headers(),
        )

        if response_scoring.status_code == 405:
            raise WMLClientError(
                "POST is not supported using this method. "
                "Send requests directly to the deployed ai_service. "
                f"Error: {response_scoring.status_code}. {response_scoring.text}"
            )

        return self._handle_response(200, "AI Service run", response_scoring)

    async def arun_ai_service(
        self,
        deployment_id: str,
        ai_service_payload: dict,
        path_suffix: str | None = None,
    ) -> Any:
        """Execute an AI service by providing a scoring payload asynchronously.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :param path_suffix: path suffix to be appended to the scoring url, defaults to None
        :type path_suffix: str, optional

        :return: response of the AI service
        :rtype: Any

        .. note::
            * By executing this class method, a POST request is performed.
            * In case of `method not allowed` error, try sending requests directly to your deployed ai service.
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        url = self._client._href_definitions.get_deployment_ai_service_href(
            deployment_id
        )
        if path_suffix is not None:
            url += f"/{path_suffix}"

        response_scoring = await self._client.async_httpx_client.post(
            url=url,
            json=ai_service_payload,
            params=self._client._params(
                skip_for_create=True, skip_userfs=True
            ),  # version parameter is mandatory
            headers=await self._client._aget_headers(),
        )

        if response_scoring.status_code == 405:
            raise WMLClientError(
                "POST is not supported using this method. "
                "Send requests directly to the deployed ai_service. "
                f"Error: {response_scoring.status_code}. {response_scoring.text}"
            )

        return self._handle_response(200, "AI Service run", response_scoring)

    def run_ai_service_stream(
        self,
        deployment_id: str,
        ai_service_payload: dict,
    ) -> Generator:
        """Execute an AI service by providing a scoring payload.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :return: stream of the response of the AI service
        :rtype: Generator
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        with self._client.httpx_client.stream(
            url=self._client._href_definitions.get_deployment_ai_service_stream_href(
                deployment_id
            ),
            json=ai_service_payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            method="POST",
        ) as resp:
            if resp.status_code == 200:
                for chunk in resp.iter_lines():
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data":
                        yield response
            else:
                resp.read()
                raise ApiRequestFailure("Failure during AI Service run stream", resp)

    async def arun_ai_service_stream(
        self,
        deployment_id: str,
        ai_service_payload: dict,
    ) -> AsyncGenerator:
        """Execute an AI service by providing a scoring payload asynchronously.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :return: stream of the response of the AI service
        :rtype: Generator
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        async with self._client.async_httpx_client.stream(
            url=self._client._href_definitions.get_deployment_ai_service_stream_href(
                deployment_id
            ),
            json=ai_service_payload,
            headers=await self._client._aget_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            method="POST",
        ) as resp:
            if resp.status_code == 200:
                for chunk in resp.iter_lines():
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data":
                        yield response
            else:
                resp.read()
                raise ApiRequestFailure("Failure during AI Service run stream", resp)


class RuntimeContext:
    """
    Class included to keep the interface compatible with the Deployment's RuntimeContext
    used in AIServices implementation.

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient

    :param request_payload_json: Request payload for testing of generate/ generate_stream call of AI Service.
    :type request_payload_json: dict, optional

    :param method: HTTP request method for testing of generate/ generate_stream call of AI Service.
    :type method: str, optional

    :param path: Request endpoint path for testing of generate/ generate_stream call of AI Service.
    :type path: str, optional

    ``
    RuntimeContext`` initialized for testing purposes before deployment:

    .. code-block:: python

        context = RuntimeContext(api_client=client, request_payload_json={"field": "value"})

    Examples of ``RuntimeContext`` usage within AI Service source code:


    .. code-block:: python

        def deployable_ai_service(context, **custom):
            task_token = context.generate_token()

            def generate(context) -> dict:
                user_token = context.get_token()
                headers = context.get_headers()
                json_body = context.get_json()
                ...
                return {"body": json_body}

            return generate

        generate = deployable_ai_service(context)
        generate_output = generate(context)  # returns {"body": {"field": "value"}}


    Change the JSON body in ``RuntimeContext``:

    .. code-block:: python

        context.request_payload_json = {"field2": "value2"}

        generate = deployable_ai_service(context)
        generate_output = generate(context)  # returns {"body": {"field2": "value2"}}
    """

    def __init__(
        self,
        api_client: APIClient,
        request_payload_json: dict | None = None,
        method: str | None = None,
        path: str | None = None,
    ):
        self._api_client = api_client
        self.request_payload_json = request_payload_json
        self.method = method
        self.path = path

    @property
    def request_payload_json(self) -> dict | None:
        return self._request_payload_json

    @request_payload_json.setter
    def request_payload_json(self, value: dict | None) -> None:
        try:
            json_value = json.loads(json.dumps(value))
        except TypeError as e:
            raise InvalidValue("request_payload_json", reason=str(e))

        self._request_payload_json = json_value

    def get_token(self) -> str:
        """Return user token."""
        return self.generate_token()

    def generate_token(self) -> str:
        """Return refreshed token."""
        return self._api_client._get_icptoken()

    def get_headers(self) -> dict:
        """Return headers with refreshed token."""
        return self._api_client._get_headers()

    def get_json(self) -> dict | None:
        """Get payload JSON send in body of API request to the generate or generate_stream method in deployed AIService.
        For testing purposes the payload JSON need to be set in RuntimeContext initialization
        or later as request_payload_json property.
        """
        return self.request_payload_json

    def get_space_id(self) -> str:
        """Return default space id."""
        return self._api_client.default_space_id

    def get_method(self) -> str:
        """Return the HTTP request method: 'GET', 'POST', etc."""
        return self.method or ""

    def get_path_suffix(self) -> str:
        """Return the suffix of ai_service endpoint including the query parameters."""
        try:
            suffix = self.path.split("ai_service", 1)[1]
        except IndexError as e:
            raise ValueError(
                "Couldn't find the path suffix since endpoint URL is incorrect."
            ) from e
        if suffix:
            suffix = suffix.removeprefix("/")
        return suffix

    def get_query_parameters(self) -> dict:
        """Return the query parameters from the ai_service endpoint as a dict."""
        parsed_url = urlparse(self.path)
        query = parsed_url.query
        params = parse_qs(query)
        if params:
            flat_params = {k: v[0] for k, v in params.items()}
            return flat_params
        else:
            return {}

    def get_bytes(self) -> bytes:
        """Return the request data as bytes."""
        payload_json = self.get_json()
        payload_str = json.dumps(payload_json)
        bytes_data = payload_str.encode("utf-8")
        return bytes_data
