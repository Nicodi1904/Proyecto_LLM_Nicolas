#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import ast
import gzip
import inspect
import os
import re
import shutil
import textwrap
import types
import uuid
from itertools import zip_longest
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
    cast,
)
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.metanames import AIServiceMetaNames
from ibm_watsonx_ai.utils import is_of_python_basic_type
from ibm_watsonx_ai.utils.utils import AsyncFileReader
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    UnexpectedType,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import pandas

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.lifecycle import SpecStates


_OAPI = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
    "tuple": "array",
}


class AIServices(WMLResource):
    """Store and manage an AI service."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

        self.ConfigurationMetaNames = AIServiceMetaNames()

    def _prepare_ai_service_metadata(
        self, meta_props: dict[str, Any], ai_service: str | Callable
    ) -> dict:
        ai_service_metadata = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        docs = ai_service_metadata.setdefault("documentation", {})

        should_doc = (
            self._client.CLOUD_PLATFORM_SPACES or self._client.CPD_version > 5.2
        )
        need_funcs = should_doc and not docs.get("functions")
        need_init = should_doc and not docs.get("init")

        if need_funcs or need_init:
            if isinstance(ai_service, str):
                with gzip.open(ai_service, "rt", encoding="utf-8") as f:
                    source = f.read()
            elif callable(ai_service):
                source = inspect.getsource(ai_service)
            else:
                raise TypeError("Argument must be a .py.gz filepath or a callable")

            if need_funcs:
                extracted_return_names = self._extract_first_return_names(source or "")
                docs["functions"] = {
                    "generate": "generate" in extracted_return_names,
                    "generate_stream": "generate_stream" in extracted_return_names,
                    "generate_batch": "generate_batch" in extracted_return_names,
                }

            if need_init:
                schema = self._extract_parameters_schema(source or "")
                docs["init"] = schema

        # at least one is set since _check_if_either_is_set() passed
        if self._client.default_space_id is not None:
            ai_service_metadata["space_id"] = self._client.default_space_id
        else:
            ai_service_metadata["project_id"] = self._client.default_project_id

        return ai_service_metadata

    def store(self, ai_service: str | Callable, meta_props: dict[str, Any]) -> dict:
        """Create an AI service asset.

        .. note::

            Supported for IBM watsonx.ai for IBM Cloud and IBM watsonx.ai software with IBM Cloud Pak® for Data (version 5.1.1 and later).

        You can use one of the following as an `ai_service`:
            - filepath to gz file
            - generator function that takes no argument or arguments that all have primitive python default values, and returns a `generate` function.

        :param ai_service: path to a file with an archived AI service function's content or a generator function (as described above)
        :type ai_service: str | Callable

        :param meta_props: metadata for storing an AI service asset. To see available meta names
            use ``client._ai_services.ConfigurationMetaNames.show()`` or direct to :class:`~metanames.AIServiceMetaNames` class.
        :type meta_props: dict

        :return: metadata of the stored AI service
        :rtype: dict

        **Examples:**

        The most simple use of an AI service is:

        .. code-block:: python

            documentation_request = {
                "application/json": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "parameters": {
                            "properties": {
                                "max_new_tokens": {"type": "integer"},
                                "top_p": {"type": "number"},
                            },
                            "required": ["max_new_tokens", "top_p"],
                        },
                    },
                    "required": ["query"],
                }
            }

            documentation_response = {
                "application/json": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "result": {"type": "string"}
                    },
                    "required": ["query", "result"],
                }
            }

            meta_props = {
                client._ai_services.ConfigurationMetaNames.NAME: "AI service example",
                client._ai_services.ConfigurationMetaNames.DESCRIPTION: "This is AI service function",
                client._ai_services.ConfigurationMetaNames.SOFTWARE_SPEC_ID: "53dc4cf1-252f-424b-b52d-5cdd9814987f",
                client._ai_services.ConfigurationMetaNames.DOCUMENTATION_REQUEST: documentation_request,
                client._ai_services.ConfigurationMetaNames.DOCUMENTATION_RESPONSE: documentation_response,
                }

            def deployable_ai_service(context, params={"k1":"v1"}, **kwargs):

                # imports
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference

                task_token = context.generate_token()

                outer_context = context
                url = "https://us-south.ml.cloud.ibm.com"
                project_id = "53dc4cf1-252f-424b-b52d-5cdd9814987f"

                def generate(context):
                    task_token = outer_context.generate_token()
                    payload = context.get_json()

                    model = ModelInference(
                        model_id="google/flan-t5-xl",
                        credentials=Credentials(
                                        url=url,
                                        token=task_token
                                        ),
                        project_id=project_id)

                    response = model.generate_text(payload['query'])
                    response_body = {'query': payload['query'],
                                     'result': response}

                    return {'body': response_body}

                return generate

            stored_ai_service_details = client._ai_services.store(deployable_ai_service, meta_props)

        """
        self._client._check_if_either_is_set()

        AIServices._validate_type(
            ai_service, "ai_service", [str, types.FunctionType], True
        )
        AIServices._validate_type(meta_props, "meta_props", dict, True)

        self.ConfigurationMetaNames._validate(meta_props)

        content_path, _, archive_name = self._prepare_ai_service_function_content(
            ai_service
        )

        try:
            ai_service_metadata = self._prepare_ai_service_metadata(
                meta_props, ai_service
            )

            response_post = requests.post(
                self._client._href_definitions.get_ai_services_href(),
                json=ai_service_metadata,
                params=self._client._params(skip_for_create=True),
                headers=self._client._get_headers(),
            )

            details = self._handle_response(
                expected_status_code=201,
                operationName="store",
                response=response_post,
            )

            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    self._client._href_definitions.get_ai_service_code_href(
                        details["metadata"]["id"]
                    ),
                    data=data,
                    params=self._client._params(),
                    headers=self._client._get_headers(no_content_type=True),
                )

        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

        if response_definition_put.status_code != 201:
            self.delete(details["metadata"]["id"])
        self._handle_response(
            201,
            "uploading AI service content",
            response_definition_put,
            json_response=False,
        )

        return details

    async def astore(
        self, ai_service: str | Callable, meta_props: dict[str, Any]
    ) -> dict:
        """Create an AI service asset asynchronously.

        .. note::

            Supported for IBM watsonx.ai for IBM Cloud and IBM watsonx.ai software with IBM Cloud Pak® for Data (version 5.1.1 and later).

        You can use one of the following as an `ai_service`:
            - filepath to gz file
            - generator function that takes no argument or arguments that all have primitive python default values, and returns a `generate` function.

        :param ai_service: path to a file with an archived AI service function's content or a generator function (as described above)
        :type ai_service: str | Callable

        :param meta_props: metadata for storing an AI service asset. To see available meta names
            use ``client._ai_services.ConfigurationMetaNames.show()`` or direct to :class:`~metanames.AIServiceMetaNames` class.
        :type meta_props: dict

        :return: metadata of the stored AI service
        :rtype: dict

        **Examples:**

        The most simple use of an AI service is:

        .. code-block:: python

            documentation_request = {
                "application/json": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "parameters": {
                            "properties": {
                                "max_new_tokens": {"type": "integer"},
                                "top_p": {"type": "number"},
                            },
                            "required": ["max_new_tokens", "top_p"],
                        },
                    },
                    "required": ["query"],
                }
            }

            documentation_response = {
                "application/json": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "result": {"type": "string"}
                    },
                    "required": ["query", "result"],
                }
            }

            meta_props = {
                client._ai_services.ConfigurationMetaNames.NAME: "AI service example",
                client._ai_services.ConfigurationMetaNames.DESCRIPTION: "This is AI service function",
                client._ai_services.ConfigurationMetaNames.SOFTWARE_SPEC_ID: "53dc4cf1-252f-424b-b52d-5cdd9814987f",
                client._ai_services.ConfigurationMetaNames.DOCUMENTATION_REQUEST: documentation_request,
                client._ai_services.ConfigurationMetaNames.DOCUMENTATION_RESPONSE: documentation_response,
                }

            def deployable_ai_service(context, params={"k1":"v1"}, **kwargs):

                # imports
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference

                task_token = context.generate_token()

                outer_context = context
                url = "https://us-south.ml.cloud.ibm.com"
                project_id = "53dc4cf1-252f-424b-b52d-5cdd9814987f"

                def generate(context):
                    task_token = outer_context.generate_token()
                    payload = context.get_json()

                    model = ModelInference(
                        model_id="google/flan-t5-xl",
                        credentials=Credentials(
                                        url=url,
                                        token=task_token
                                        ),
                        project_id=project_id)

                    response = model.generate_text(payload['query'])
                    response_body = {'query': payload['query'],
                                     'result': response}

                    return {'body': response_body}

                return generate

            stored_ai_service_details = await client._ai_services.astore(deployable_ai_service, meta_props)

        """
        self._client._check_if_either_is_set()

        AIServices._validate_type(
            ai_service, "ai_service", [str, types.FunctionType], True
        )
        AIServices._validate_type(meta_props, "meta_props", dict, True)

        self.ConfigurationMetaNames._validate(meta_props)

        content_path, _, archive_name = self._prepare_ai_service_function_content(
            ai_service
        )

        try:
            ai_service_metadata = self._prepare_ai_service_metadata(
                meta_props, ai_service
            )

            response_post = await self._client.async_httpx_client.post(
                self._client._href_definitions.get_ai_services_href(),
                json=ai_service_metadata,
                params=self._client._params(skip_for_create=True),
                headers=await self._client._aget_headers(),
            )

            details = self._handle_response(
                expected_status_code=201,
                operationName="store",
                response=response_post,
            )

            response_definition_put = await self._client.async_httpx_client.put(
                self._client._href_definitions.get_ai_service_code_href(
                    details["metadata"]["id"]
                ),
                content=AsyncFileReader(content_path),
                params=self._client._params(),
                headers=await self._client._aget_headers(no_content_type=True),
            )

        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

        if response_definition_put.status_code != 201:
            await self.adelete(details["metadata"]["id"])

        self._handle_response(
            201,
            "uploading AI service content",
            response_definition_put,
            json_response=False,
        )

        return details

    def update(
        self,
        ai_service_id: str,
        changes: dict,
        update_ai_service: str | Callable | None = None,
    ) -> dict:
        """Updates existing AI service asset metadata.

        :param ai_service_id: ID of AI service to be updated
        :type ai_service_id: str

        :param changes: elements that will be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :param update_ai_service: path to the file with an archived AI service function's content or function that will be changed for a specific ai_service_id
        :type update_ai_service: str | Callable, optional

        **Example:**

        .. code-block:: python

            metadata = {
                client._ai_services.ConfigurationMetaNames.NAME: "updated_ai_service"
            }

            ai_service_details = client._ai_services.update(ai_service_id, changes=metadata)

        """

        self._client._check_if_either_is_set()

        self._validate_type(ai_service_id, "ai_service_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        changes = cast(dict, changes)

        details = self.get_details(ai_service_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, changes, with_validation=True
        )

        response = requests.patch(
            self._client._href_definitions.get_ai_service_href(ai_service_id),
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        updated_details = self._handle_response(200, "AI service patch", response)

        if update_ai_service is not None:
            self._update_ai_service_content(ai_service_id, update_ai_service)

        return updated_details

    async def aupdate(
        self,
        ai_service_id: str,
        changes: dict,
        update_ai_service: str | Callable | None = None,
    ) -> dict:
        """Updates existing AI service asset metadata asynchronously.

        :param ai_service_id: ID of AI service to be updated
        :type ai_service_id: str

        :param changes: elements that will be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :param update_ai_service: path to the file with an archived AI service function's content or function that will be changed for a specific ai_service_id
        :type update_ai_service: str | Callable, optional

        **Example:**

        .. code-block:: python

            metadata = {
                client._ai_services.ConfigurationMetaNames.NAME: "updated_ai_service"
            }

            ai_service_details = await client._ai_services.aupdate(ai_service_id, changes=metadata)

        """

        self._client._check_if_either_is_set()

        self._validate_type(ai_service_id, "ai_service_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        changes = cast(dict, changes)

        details = await self.aget_details(ai_service_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, changes, with_validation=True
        )

        response = await self._client.async_httpx_client.patch(
            self._client._href_definitions.get_ai_service_href(ai_service_id),
            json=patch_payload,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )
        updated_details = self._handle_response(200, "AI service patch", response)

        if update_ai_service is not None:
            await self._aupdate_ai_service_content(ai_service_id, update_ai_service)

        return updated_details

    def _update_ai_service_content(
        self,
        ai_service_id: str,
        updated_function: str | Callable,
    ) -> None:
        AIServices._validate_type(
            updated_function, "updated_function", [str, types.FunctionType], True
        )

        ai_service_id = cast(str, ai_service_id)

        updated_function = cast(str | Callable, updated_function)
        content_path, _, archive_name = self._prepare_ai_service_function_content(
            updated_function
        )
        try:
            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    self._client._href_definitions.get_ai_service_code_href(
                        ai_service_id
                    ),
                    data=data,
                    params=self._client._params(),
                    headers=self._client._get_headers(no_content_type=True),
                )
                if response_definition_put.status_code != 201:
                    raise WMLClientError(
                        "Unable to update AI service content "
                        + str(response_definition_put.content)
                    )
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

    async def _aupdate_ai_service_content(
        self,
        ai_service_id: str,
        updated_function: str | Callable,
    ) -> None:
        AIServices._validate_type(
            updated_function, "updated_function", [str, types.FunctionType], True
        )

        ai_service_id = cast(str, ai_service_id)

        updated_function = cast(str | Callable, updated_function)
        content_path, _, archive_name = self._prepare_ai_service_function_content(
            updated_function
        )
        try:
            response_definition_put = await self._client.async_httpx_client.put(
                self._client._href_definitions.get_ai_service_code_href(ai_service_id),
                content=AsyncFileReader(content_path),
                params=self._client._params(),
                headers=await self._client._aget_headers(no_content_type=True),
            )
            if response_definition_put.status_code != 201:
                raise WMLClientError(
                    "Unable to update AI service content "
                    + str(response_definition_put.content)
                )
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

    def download(
        self,
        ai_service_id: str,
        filename: str = "downloaded_ai_service_function.gz",
        rev_id: str | None = None,
    ) -> str:
        """Download an AI service’s content from a watsonx.ai repository to a local file.

        :param ai_service_id: stored AI service ID
        :type ai_service_id: str

        :param filename: name of the local file to be created, example: ai_service_content.py.gz
        :type filename: str, optional

        :param rev_id: revision ID
        :type rev_id: str, optional

        :return: path to the downloaded AI service content
        :rtype: str

        **Example:**

        .. code-block:: python

            client._ai_services.download(ai_service_id, 'my_ai_service.py.gz')

        """

        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)
        AIServices._validate_type(filename, "filename", str, True)

        artifact_content_url = self._client._href_definitions.get_ai_service_code_href(
            ai_service_id
        )

        try:
            params = self._client._params()
            if rev_id is not None:
                params.update({"rev": rev_id})

            response = requests.get(
                artifact_content_url,
                params=params,
                headers=self._client._get_headers(),
                stream=True,
            )
            if response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during downloading AI service.", response
                )

            downloaded_model = response.content
            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s",
                artifact_content_url,
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(
                f"Downloading function content with artifact_url: '{artifact_content_url}' failed.",
                e,
            )

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)
            print(f"Successfully saved AI service content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving AI service content with artifact_url: '{filename}' failed.",
                e,
            )

    async def adownload(
        self,
        ai_service_id: str,
        filename: str = "downloaded_ai_service_function.gz",
        rev_id: str | None = None,
    ) -> str:
        """Download an AI service’s content from a watsonx.ai repository to a local file asynchronously.

        :param ai_service_id: stored AI service ID
        :type ai_service_id: str

        :param filename: name of the local file to be created, example: ai_service_content.py.gz
        :type filename: str, optional

        :param rev_id: revision ID
        :type rev_id: str, optional

        :return: path to the downloaded AI service content
        :rtype: str

        **Example:**

        .. code-block:: python

            await client._ai_services.adownload(ai_service_id, 'my_ai_service.py.gz')

        """

        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)
        AIServices._validate_type(filename, "filename", str, True)

        artifact_content_url = self._client._href_definitions.get_ai_service_code_href(
            ai_service_id
        )

        try:
            params = self._client._params()
            if rev_id is not None:
                params.update({"rev": rev_id})

            async with self._client.async_httpx_client.stream(
                method="GET",
                url=artifact_content_url,
                params=params,
                headers=await self._client._aget_headers(),
            ) as response:
                if response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during downloading AI service.", response
                    )

                downloaded_model = await response.aread()

            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s",
                artifact_content_url,
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(
                f"Downloading function content with artifact_url: '{artifact_content_url}' failed.",
                e,
            )

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)
            print(f"Successfully saved AI service content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving AI service content with artifact_url: '{filename}' failed.",
                e,
            )

    def delete(self, ai_service_id: str, force: bool = False) -> Literal["SUCCESS"]:
        """Delete a stored AI service asset.

        :param ai_service_id: stored AI service ID
        :type ai_service_id: str

        :param force: if True, the delete operation will proceed even when the AI service deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client._ai_services.delete(ai_service_id)
        """
        self._client._check_if_either_is_set()
        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)

        if not force and self._if_deployment_exist_for_asset(ai_service_id):
            raise WMLClientError(
                "Cannot delete AI service that has existing deployments. Please delete all associated deployments and try again"
            )

        ai_service_endpoint = self._client._href_definitions.get_ai_service_href(
            ai_service_id
        )

        self._logger.debug(
            "Deletion artifact AI service endpoint: %s", ai_service_endpoint
        )
        response = requests.delete(
            ai_service_endpoint,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "AI service deletion", response, False)

    async def adelete(
        self, ai_service_id: str, force: bool = False
    ) -> Literal["SUCCESS"]:
        """Delete a stored AI service asset asynchronously.

        :param ai_service_id: stored AI service ID
        :type ai_service_id: str

        :param force: if True, the delete operation will proceed even when the AI service deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client._ai_services.adelete(ai_service_id)
        """
        self._client._check_if_either_is_set()
        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)

        if not force and await self._aif_deployment_exist_for_asset(ai_service_id):
            raise WMLClientError(
                "Cannot delete AI service that has existing deployments. Please delete all associated deployments and try again"
            )

        ai_service_endpoint = self._client._href_definitions.get_ai_service_href(
            ai_service_id
        )

        self._logger.debug(
            "Deletion artifact AI service endpoint: %s", ai_service_endpoint
        )
        response = await self._client.async_httpx_client.delete(
            ai_service_endpoint,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(204, "AI service deletion", response, False)

    def get_details(
        self,
        ai_service_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        ai_service_name: str | None = None,
    ) -> dict | Generator:
        """Get the metadata of AI service(s). If neither AI service ID nor AI service name is specified,
        all AI service metadata is returned.
        If only AI service name is specified, metadata of AI services with the name is returned (if any).

        :param ai_service_id: ID of the AI service
        :type ai_service_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int | None, optional

        :param asynchronous: if `True`, it will work as a generator, defaults to False
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks, defaults to False
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `ai_service_id` is None
        :type spec_state: SpecStates | None, optional

        :param ai_service_name: name of the AI service, can be used only when `ai_service_id` is None
        :type ai_service_name: str, optional

        :return: metadata of the AI service
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In the current implementation setting, `spec_state=True` might break the set `limit` and return less records than stated in the set `limit`.

        **Examples:**

        .. code-block:: python

            ai_service_details = client._ai_services.get_details(ai_service_id)
            ai_service_details = client._ai_services.get_details(ai_service_name)
            ai_service_details = client._ai_services.get_details()
            ai_service_details = client._ai_services.get_details(limit=100)
            ai_service_details = client._ai_services.get_details(limit=100, get_all=True)
            ai_service_details = []
            for entry in client._ai_services.get_details(limit=100, asynchronous=True, get_all=True):
                ai_service_details.extend(entry)

        """

        AIServices._validate_type(ai_service_id, "ai_service_id", str, False)
        AIServices._validate_type(limit, "limit", int, False)
        AIServices._validate_type(asynchronous, "asynchronous", bool, False)
        AIServices._validate_type(get_all, "get_all", bool, False)
        AIServices._validate_type(spec_state, "spec_state", object, False)

        if limit and spec_state:
            spec_state_setting_warning = (
                "Warning: In current implementation setting `spec_state=ibm_watsonx_ai.lifecycle.SUPPORTED` may break "
                "set `limit`, returning less records than stated by set `limit`."
            )
            warn(spec_state_setting_warning)

        self._client._check_if_either_is_set()
        url = self._client._href_definitions.get_ai_services_href()

        if ai_service_id is not None:
            return self._get_artifact_details(
                url, ai_service_id, limit, "AI services", _all=get_all
            )

        if spec_state:
            filter_func = self._get_filter_func_by_spec_ids(
                self._get_and_cache_spec_ids_for_state(spec_state)
            )
        elif ai_service_name:
            filter_func = self._get_filter_func_by_artifact_name(ai_service_name)
        else:
            filter_func = None

        return self._get_artifact_details(
            url,
            ai_service_id,
            limit,
            "AI services",
            _async=asynchronous,
            _all=get_all,
            _filter_func=filter_func,
        )

    async def aget_details(
        self,
        ai_service_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        ai_service_name: str | None = None,
    ) -> dict | AsyncGenerator:
        """Get the metadata of AI service(s) asynchronously. If neither AI service ID nor AI service name is specified,
        all AI service metadata is returned.
        If only AI service name is specified, metadata of AI services with the name is returned (if any).

        :param ai_service_id: ID of the AI service
        :type ai_service_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int | None, optional

        :param asynchronous: if `True`, it will work as a generator, defaults to False
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks, defaults to False
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `ai_service_id` is None
        :type spec_state: SpecStates | None, optional

        :param ai_service_name: name of the AI service, can be used only when `ai_service_id` is None
        :type ai_service_name: str, optional

        :return: metadata of the AI service
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In the current implementation setting, `spec_state=True` might break the set `limit` and return less records than stated in the set `limit`.

        **Examples:**

        .. code-block:: python

            ai_service_details = await client._ai_services.aget_details(ai_service_id)
            ai_service_details = await client._ai_services.aget_details(ai_service_name)
            ai_service_details = await client._ai_services.aget_details()
            ai_service_details = await client._ai_services.aget_details(limit=100)
            ai_service_details = await client._ai_services.aget_details(limit=100, get_all=True)
            ai_service_details = []
            async for entry in await client._ai_services.aget_details(limit=100, asynchronous=True, get_all=True):
                ai_service_details.extend(entry)

        """

        AIServices._validate_type(ai_service_id, "ai_service_id", str, False)
        AIServices._validate_type(limit, "limit", int, False)
        AIServices._validate_type(asynchronous, "asynchronous", bool, False)
        AIServices._validate_type(get_all, "get_all", bool, False)
        AIServices._validate_type(spec_state, "spec_state", object, False)

        if limit and spec_state:
            spec_state_setting_warning = (
                "Warning: In current implementation setting `spec_state=ibm_watsonx_ai.lifecycle.SUPPORTED` may break "
                "set `limit`, returning less records than stated by set `limit`."
            )
            warn(spec_state_setting_warning)

        self._client._check_if_either_is_set()
        url = self._client._href_definitions.get_ai_services_href()

        if ai_service_id is not None:
            return await self._aget_artifact_details(
                url, ai_service_id, limit, "AI services", _all=get_all
            )

        if spec_state:
            filter_func = self._get_filter_func_by_spec_ids(
                self._get_and_cache_spec_ids_for_state(spec_state)
            )
        elif ai_service_name:
            filter_func = self._get_filter_func_by_artifact_name(ai_service_name)
        else:
            filter_func = None

        return await self._aget_artifact_details(
            url,
            ai_service_id,
            limit,
            "AI services",
            _async=asynchronous,
            _all=get_all,
            _filter_func=filter_func,
        )

    @classmethod
    def get_id(cls, ai_service_details: dict) -> str:
        """Get the ID of a stored AI service.

        :param ai_service_details: metadata of the stored AI service
        :type ai_service_details: dict

        :return: ID of the stored AI service
        :rtype: str

        **Example:**

        .. code-block:: python

            ai_service_details = client.repository.get_ai_service_details(ai_service_id)
            ai_service_id = client._ai_services.get_id(ai_service_details)
        """

        cls._validate_type(ai_service_details, "ai_service_details", dict, True)
        return cls._get_required_element_from_dict(
            ai_service_details, "ai_service_details", ["metadata", "id"]
        )

    def list(self, limit: int | None = None) -> pandas.DataFrame:
        """Return stored AI services in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed AI services
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._ai_services.list()
        """
        self._client._check_if_either_is_set()

        ai_services_resources = self.get_details(
            get_all=self._should_get_all_values(limit)
        )["resources"]

        ai_service_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
                m["entity"]["code_type"],
                self._client.software_specifications._get_state(m),
                self._client.software_specifications._get_replacement(m),
            )
            for m in ai_services_resources
        ]

        return self._list(
            ai_service_values,
            ["ID", "NAME", "CREATED", "TYPE", "SPEC_STATE", "SPEC_REPLACEMENT"],
            limit,
        )

    def create_revision(self, ai_service_id: str) -> dict:
        """Create a new AI service revision.

        :param ai_service_id: unique ID of the AI service
        :type ai_service_id: str

        :return: revised metadata of the stored AI service
        :rtype: dict

        **Example:**

        .. code-block:: python

            client._ai_services.create_revision(ai_service_id)
        """

        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)

        return self._create_revision_artifact(
            self._client._href_definitions.get_ai_services_href(),
            ai_service_id,
            "ai_service_id",
        )

    async def acreate_revision(self, ai_service_id: str) -> dict:
        """Create a new AI service revision asynchronously.

        :param ai_service_id: unique ID of the AI service
        :type ai_service_id: str

        :return: revised metadata of the stored AI service
        :rtype: dict

        **Example:**

        .. code-block:: python

            await client._ai_services.acreate_revision(ai_service_id)
        """

        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)

        return await self._acreate_revision_artifact(
            self._client._href_definitions.get_ai_services_href(),
            ai_service_id,
            "ai_service_id",
        )

    def get_revision_details(
        self,
        ai_service_id: str,
        rev_id: str,
    ) -> dict:
        """Get the metadata of a specific revision of a stored AI service.

        :param ai_service_id: definition of the stored AI service
        :type ai_service_id: str

        :param rev_id: unique ID of the AI service revision
        :type rev_id: str

        :return: metadata of the stored AI service revision
        :rtype: dict

        **Example:**

        .. code-block:: python

            ai_service_revision_details = client._ai_services.get_revision_details(ai_service_id, rev_id)

        """

        self._client._check_if_either_is_set()
        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)
        AIServices._validate_type(rev_id, "rev_id", str, True)

        return self._get_with_or_without_limit(
            self._client._href_definitions.get_ai_service_href(ai_service_id),
            limit=None,
            op_name="AI service",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    async def aget_revision_details(
        self,
        ai_service_id: str,
        rev_id: str,
    ) -> dict:
        """Get the metadata of a specific revision of a stored AI service asynchronously.

        :param ai_service_id: definition of the stored AI service
        :type ai_service_id: str

        :param rev_id: unique ID of the AI service revision
        :type rev_id: str

        :return: metadata of the stored AI service revision
        :rtype: dict

        **Example:**

        .. code-block:: python

            ai_service_revision_details = await client._ai_services.aget_revision_details(ai_service_id, rev_id)

        """

        self._client._check_if_either_is_set()
        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)
        AIServices._validate_type(rev_id, "rev_id", str, True)

        return await self._aget_with_or_without_limit(
            self._client._href_definitions.get_ai_service_href(ai_service_id),
            limit=None,
            op_name="AI service",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def list_revisions(
        self, ai_service_id: str, limit: int | None = None
    ) -> pandas.DataFrame:
        """Print all revisions for a given AI service ID in a table format.

        :param ai_service_id: unique ID of the stored AI service
        :type ai_service_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._ai_services.list_revisions(ai_service_id)

        """
        self._client._check_if_either_is_set()

        AIServices._validate_type(ai_service_id, "ai_service_id", str, True)

        ai_services_details = self._get_artifact_details(
            self._client._href_definitions.get_ai_service_revisions_href(ai_service_id),
            None,
            None,
            "AI service revisions",
            _all=self._should_get_all_values(limit),
        )

        ai_service_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["rev"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in ai_services_details["resources"]
        ]

        return self._list(
            ai_service_values,
            ["ID", "REV", "NAME", "CREATED"],
            limit,
        )

    @staticmethod
    def _prepare_ai_service_function_content(
        ai_service_function: str | Callable,
    ) -> tuple[str, bool, str | None]:
        """Prepare the content of an AI service function for storing in the repository.
        If a Callable is passed, the function creates an archive.

        :param ai_service_function: path to a file with an archived AI service function’s content or a generator function
        :type ai_service_function: str | Callable

        :raises UnexpectedType: if any of the ``ai_service_function`` default parameters is not of basic Python type

        :raises WMLClientError: if ``ai_service_function`` is defined incorrectly

        :return: path to the compressed AI service function source if the archive is provided by the user, name of the archive if not provided by the user
        :rtype: tuple[str, bool, str | None]
        """
        if isinstance(ai_service_function, str):
            return ai_service_function, True, None

        user_content_file = False
        archive_name = None

        try:
            code = AIServices._populate_default_params(
                ai_service_function=ai_service_function
            )

            tmp_uid = f"tmp_ai_service_python_function_code_{str(uuid.uuid4()).replace('-', '_')}"
            filename = f"{tmp_uid}.py"

            with open(filename, "w") as f:
                f.write(code)

            archive_name = f"{tmp_uid}.py.gz"

            with open(filename, "rb") as f_in:
                with gzip.open(archive_name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(filename)

            return archive_name, user_content_file, archive_name

        except Exception as e:
            try:
                os.remove(filename)
            except Exception:
                pass
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass
            raise WMLClientError("Exception during getting AI service content code.", e)

    @staticmethod
    def _populate_default_params(
        ai_service_function: Callable, _validate_values: bool = True
    ) -> str:
        code_lines = inspect.getsource(ai_service_function)
        code = textwrap.dedent(code_lines or "")

        signature = inspect.signature(ai_service_function)

        # check all except the first parameter, as it should be `context` and not have a default value
        if _validate_values:
            for param, value in list(signature.parameters.items())[1:]:
                if (
                    value.default is not inspect.Parameter.empty
                    and not is_of_python_basic_type(value.default)
                ):
                    raise UnexpectedType(
                        param, "primitive python type", type(value.default)
                    )

        args_list = list(signature.parameters.keys())
        patterns = []
        type_hint_pattern = r"(?:[:]\s*[\S\[\]\s]+\s*?)?"
        # all but last -> expect a trailing comma in the signature
        for arg in args_list[:-1]:
            p = signature.parameters[arg]
            if p.default is inspect.Parameter.empty:
                patterns.append(rf"\s*{arg}{type_hint_pattern}\s*")
            else:
                patterns.append(rf"\s*{arg}\s*{type_hint_pattern}\s*=\s*([\s\S]+?)\s*")

        # last arg -> allow optional comma before the ')'
        last = args_list[-1]
        p_last = signature.parameters[last]
        if p_last.default is inspect.Parameter.empty:
            patterns.append(rf"\s*(?:\*\*)?{last}{type_hint_pattern}\s*(?:,\s*)?")
        else:
            patterns.append(rf"\s*{last}\s*{type_hint_pattern}\s*=\s*([\s\S]+?)\s*")

        args_pattern = ",".join(patterns)
        regex = re.compile(
            rf"^def {ai_service_function.__name__}\s*\(\s*{args_pattern}\s*\)\s*:",
            re.DOTALL | re.MULTILINE,
        )
        m = regex.search(code)

        if m is None:
            raise ValueError(
                "Unable to process AI service function content. "
                "Please make sure the function object has simply Python signature."
            )

        defaults = [
            param.default
            for param in signature.parameters.values()
            if param.default is not inspect.Parameter.empty
        ]
        for i in range(len(defaults) - 1, -1, -1):
            start, end = m.start(i + 1), m.end(i + 1)
            code = code[:start] + defaults[i].__repr__() + code[end:]

        return code

    @staticmethod
    def _extract_first_return_names(source: str) -> list:
        source_dedent = textwrap.dedent(source or "")
        tree = ast.parse(source_dedent)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Return) or stmt.value is None:
                    continue
                ret = stmt.value
                elts = ret.elts if isinstance(ret, ast.Tuple) else [ret]
                names = []
                for elt in elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
                    else:
                        names.append(ast.unparse(elt).strip())
                return names
            break
        return []

    def _extract_parameters_schema(self, source: str) -> dict:
        fn = self._first_functiondef(source)
        if not fn:
            return {"type": "object", "properties": {}}

        props, required = {}, []
        for p in self._iter_params(fn):
            name = p["name"]
            if name == "context":
                continue
            entry = {}
            if p.get("type"):
                entry["type"] = p["type"]
            if "default" in p:
                entry["default"] = p["default"]
            else:
                required.append(name)
            props[name] = entry

        schema = {"type": "object", "properties": props}
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _first_functiondef(source: str) -> ast.FunctionDef | None:
        source_dedent = textwrap.dedent(source or "")
        tree = ast.parse(source_dedent)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node
        return None

    def _iter_params(self, fn: ast.FunctionDef) -> dict[str, Any]:
        posonly = getattr(fn.args, "posonlyargs", []) or []
        pos = list(posonly) + list(fn.args.args)

        defaults = fn.args.defaults or []
        n_def = len(defaults)
        start_def = len(pos) - n_def

        # positional args
        for idx, a in enumerate(pos):
            item = {"name": a.arg}
            if a.annotation:
                t = self._ann_node_to_type(a.annotation)
                if t:
                    item["type"] = t
            if idx >= start_def and n_def > 0:
                dnode = defaults[idx - start_def]
                item["default"] = self._safe_eval_default(dnode)
            yield item

        # keyword args
        kwonly = fn.args.kwonlyargs or []
        kwdefs = fn.args.kw_defaults or []
        for a, dnode in zip_longest(kwonly, kwdefs, fillvalue=None):
            item = {"name": a.arg}
            if a.annotation:
                t = self._ann_node_to_type(a.annotation)
                if t:
                    item["type"] = t
            if dnode is not None:
                item["default"] = self._safe_eval_default(dnode)
            yield item

    @staticmethod
    def _safe_eval_default(node: ast.AST):
        try:
            return ast.literal_eval(node)
        except Exception:
            return ast.unparse(node)

    def _ann_node_to_type(self, node: ast.AST):
        match node:
            case ast.Name(id=name):
                return _OAPI.get(name)

            case ast.Attribute(attr=attr):
                return _OAPI.get(attr)

            case ast.Subscript(value=ast.Name(id=name), slice=s) if name in (
                "Optional",
                "Union",
            ):
                args = s.elts if isinstance(s, ast.Tuple) else [s]
                args = [
                    a
                    for a in args
                    if not (isinstance(a, ast.Name) and a.id in ("None", "NoneType"))
                ]
                return self._ann_node_to_type(args[0]) if len(args) == 1 else None

            case ast.Subscript(value=ast.Name(id=name)) if name in ("list", "dict"):
                return _OAPI.get(name)

            case _:
                return None
