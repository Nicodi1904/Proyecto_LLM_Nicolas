#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import gzip
import inspect
import os
import re
import shutil
import types
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal, cast
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import FunctionMetaNames
from ibm_watsonx_ai.utils import FUNCTION_DETAILS_TYPE, is_of_python_basic_type
from ibm_watsonx_ai.utils.utils import AsyncFileReader, _get_id_from_deprecated_uid
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


class Functions(WMLResource):
    """Store and manage functions."""

    ConfigurationMetaNames = FunctionMetaNames()

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def store(self, function: str | Callable, meta_props: str | dict[str, Any]) -> dict:
        """Create a function.

        As a 'function' may be used one of the following:
            - filepath to gz file
            - 'score' function reference, where the function is the function which will be deployed
            - generator function, which takes no argument or arguments which all have primitive python default values
              and as result return 'score' function

        :param function: path to file with archived function content or function (as described above)
        :type function: str or function
        :param meta_props: meta data or name of the function, to see available meta names
            use ``client._functions.ConfigurationMetaNames.show()``
        :type meta_props: str or dict

        :return: stored function metadata
        :rtype: dict

        **Examples**

        The most simple use is (using `score` function):

        .. code-block:: python

            meta_props = {
                client._functions.ConfigurationMetaNames.NAME: "function",
                client._functions.ConfigurationMetaNames.DESCRIPTION: "This is ai function",
                client._functions.ConfigurationMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

            def score(payload):
                values = [[row[0]*row[1]] for row in payload['values']]
                return {'fields': ['multiplication'], 'values': values}

            stored_function_details = client._functions.store(score, meta_props)

        Other, more interesting example is using generator function.
        In this situation it is possible to pass some variables:

        .. code-block:: python

            creds = {...}

            def gen_function(credentials=creds, x=2):
                def f(payload):
                    values = [[row[0]*row[1]*x] for row in payload['values']]
                    return {'fields': ['multiplication'], 'values': values}
                return f

            stored_function_details = client._functions.store(gen_function, meta_props)

        """
        self._client._check_if_either_is_set()

        Functions._validate_type(function, "function", [str, types.FunctionType], True)
        Functions._validate_type(meta_props, "meta_props", [dict, str], True)

        if isinstance(meta_props, str):
            meta_props = {self.ConfigurationMetaNames.NAME: meta_props}

        self.ConfigurationMetaNames._validate(meta_props)

        content_path, _, archive_name = self._prepare_function_content(function)

        try:
            function_metadata = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

            if self._client.default_space_id is not None:
                function_metadata["space_id"] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                function_metadata["project_id"] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. "
                    "Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed."
                )

            response_post = requests.post(
                self._client._href_definitions.get_functions_href(),
                json=function_metadata,
                params=self._client._params(skip_for_create=True),
                headers=self._client._get_headers(),
            )

            details = self._handle_response(
                expected_status_code=201,
                operationName="saving function",
                response=response_post,
            )

            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    self._client._href_definitions.get_function_code_href(
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
            201, "saving function content", response_definition_put, json_response=False
        )

        return details

    async def astore(
        self, function: str | Callable, meta_props: str | dict[str, Any]
    ) -> dict:
        """Create a function asynchronously.

        As a 'function' may be used one of the following:
            - filepath to gz file
            - 'score' function reference, where the function is the function which will be deployed
            - generator function, which takes no argument or arguments which all have primitive python default values
              and as result return 'score' function

        :param function: path to file with archived function content or function (as described above)
        :type function: str or function
        :param meta_props: meta data or name of the function, to see available meta names
            use ``client._functions.ConfigurationMetaNames.show()``
        :type meta_props: str or dict

        :return: stored function metadata
        :rtype: dict

        **Examples**

        The most simple use is (using `score` function):

        .. code-block:: python

            meta_props = {
                client._functions.ConfigurationMetaNames.NAME: "function",
                client._functions.ConfigurationMetaNames.DESCRIPTION: "This is ai function",
                client._functions.ConfigurationMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

            def score(payload):
                values = [[row[0]*row[1]] for row in payload['values']]
                return {'fields': ['multiplication'], 'values': values}

            stored_function_details = await client._functions.astore(score, meta_props)

        Other, more interesting example is using generator function.
        In this situation it is possible to pass some variables:

        .. code-block:: python

            creds = {...}

            def gen_function(credentials=creds, x=2):
                def f(payload):
                    values = [[row[0]*row[1]*x] for row in payload['values']]
                    return {'fields': ['multiplication'], 'values': values}
                return f

            stored_function_details = await client._functions.astore(gen_function, meta_props)

        """
        self._client._check_if_either_is_set()

        Functions._validate_type(function, "function", [str, types.FunctionType], True)
        Functions._validate_type(meta_props, "meta_props", [dict, str], True)

        if isinstance(meta_props, str):
            meta_props = {self.ConfigurationMetaNames.NAME: meta_props}

        self.ConfigurationMetaNames._validate(meta_props)

        content_path, _, archive_name = self._prepare_function_content(function)

        try:
            function_metadata = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

            if self._client.default_space_id is not None:
                function_metadata["space_id"] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                function_metadata["project_id"] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. "
                    "Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed."
                )

            response_post = await self._client.async_httpx_client.post(
                self._client._href_definitions.get_functions_href(),
                json=function_metadata,
                params=self._client._params(skip_for_create=True),
                headers=await self._client._aget_headers(),
            )

            details = self._handle_response(
                expected_status_code=201,
                operationName="saving function",
                response=response_post,
            )

            response_definition_put = await self._client.async_httpx_client.put(
                self._client._href_definitions.get_function_code_href(
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
            201, "saving function content", response_definition_put, json_response=False
        )

        return details

    def update(
        self,
        function_id: str | None = None,
        changes: dict | None = None,
        update_function: str | Callable | None = None,
        **kwargs,
    ) -> dict:
        """Updates existing function metadata.

        :param function_id: ID of function which define what should be updated
        :type function_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict
        :param update_function: path to file with archived function content or function which should be changed
            for specific function_id, this parameter is valid only for CP4D 3.0.0
        :type update_function: str or function, optional

        **Example:**

        .. code-block:: python

            metadata = {
                client._functions.ConfigurationMetaNames.NAME: "updated_function"
            }

            function_details = client._functions.update(function_id, changes=metadata)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        self._client._check_if_either_is_set()

        self._validate_type(function_id, "function_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self.get_details(function_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, changes, with_validation=True
        )

        response = requests.patch(
            self._client._href_definitions.get_function_href(function_id),
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        updated_details = self._handle_response(200, "function patch", response)

        if update_function is not None:
            self._update_function_content(function_id, update_function)

        return updated_details

    async def aupdate(
        self,
        function_id: str,
        changes: dict,
        update_function: str | Callable | None = None,
    ) -> dict:
        """Updates existing function metadata asynchronously.

        :param function_id: ID of function which define what should be updated
        :type function_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict
        :param update_function: path to file with archived function content or function which should be changed
            for specific function_id, this parameter is valid only for CP4D 3.0.0
        :type update_function: str or function, optional

        **Example:**

        .. code-block:: python

            metadata = {
                client._functions.ConfigurationMetaNames.NAME: "updated_function"
            }

            function_details = await client._functions.aupdate(function_id, changes=metadata)

        """

        self._client._check_if_either_is_set()

        self._validate_type(function_id, "function_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = await self.aget_details(function_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, changes, with_validation=True
        )

        response = await self._client.async_httpx_client.patch(
            self._client._href_definitions.get_function_href(function_id),
            json=patch_payload,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        updated_details = self._handle_response(200, "function patch", response)

        if update_function is not None:
            await self._aupdate_function_content(function_id, update_function)

        return updated_details

    def _update_function_content(
        self,
        function_id: str | None = None,
        updated_function: str | Callable | None = None,
        **kwargs,
    ) -> None:
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        Functions._validate_type(
            updated_function, "function", [str, types.FunctionType], True
        )

        updated_function = cast(str | Callable, updated_function)
        content_path, _, archive_name = self._prepare_function_content(updated_function)

        try:
            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    self._client._href_definitions.get_function_code_href(function_id),
                    data=data,
                    params=self._client._params(),
                    headers=self._client._get_headers(no_content_type=True),
                )

            if response_definition_put.status_code not in {200, 201, 204}:
                raise WMLClientError(
                    "Unable to update function content" + response_definition_put.text
                )
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

    async def _aupdate_function_content(
        self,
        function_id: str,
        updated_function: str | Callable,
    ) -> None:
        Functions._validate_type(
            updated_function, "function", [str, types.FunctionType], True
        )

        content_path, _, archive_name = self._prepare_function_content(updated_function)

        try:
            response_definition_put = await self._client.async_httpx_client.put(
                self._client._href_definitions.get_function_code_href(function_id),
                content=AsyncFileReader(content_path),
                params=self._client._params(),
                headers=await self._client._aget_headers(no_content_type=True),
            )

            if response_definition_put.status_code not in {200, 201, 204}:
                raise WMLClientError(
                    "Unable to update function content" + response_definition_put.text
                )
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

    def download(
        self,
        function_id: str | None = None,
        filename: str = "downloaded_function.gz",
        rev_id: str | None = None,
        **kwargs,
    ) -> str:
        """Download function content from watsonx.ai repository to local file.

        :param function_id: stored function ID
        :type function: str

        :param filename: name of local file to create, example: function_content.gz
        :type filename: str, optional

        :return: path to the downloaded function content
        :rtype: str

        **Example:**

        .. code-block:: python

            client._functions.download(function_id, 'my_func.tar.gz')

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)
        if rev_id is not None and isinstance(rev_id, int):
            rev_id_as_int_deprecated_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_deprecated_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(filename, "filename", str, True)

        artifact_url = self._client._href_definitions.get_function_href(function_id)

        artifact_content_url = self._client._href_definitions.get_function_code_href(
            function_id
        )

        try:
            params = self._client._params()

            if rev_id is not None:
                rev_param_key = (
                    "rev" if self._client.CLOUD_PLATFORM_SPACES else "revision_id"
                )
                params[rev_param_key] = rev_id

            response = requests.get(
                artifact_content_url,
                params=params,
                headers=self._client._get_headers(),
                stream=True,
            )

            if response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during downloading function.", response
                )

            downloaded_model = response.content
            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s", artifact_url
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(
                f"Downloading function content with artifact_url: '{artifact_url}' failed.",
                e,
            )

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)

            print(f"Successfully saved function content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving function content with artifact_url: '{filename}' failed.",
                e,
            )

    async def adownload(
        self,
        function_id: str,
        filename: str = "downloaded_function.gz",
        rev_id: str | None = None,
    ) -> str:
        """Download function content from watsonx.ai repository to local file asynchronously.

        :param function_id: stored function ID
        :type function: str

        :param filename: name of local file to create, example: function_content.gz
        :type filename: str, optional

        :return: path to the downloaded function content
        :rtype: str

        **Example:**

        .. code-block:: python

            await client._functions.adownload(function_id, 'my_func.tar.gz')

        """

        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(filename, "filename", str, True)

        artifact_url = self._client._href_definitions.get_function_href(function_id)

        artifact_content_url = self._client._href_definitions.get_function_code_href(
            function_id
        )

        try:
            params = self._client._params()

            if rev_id is not None:
                rev_param_key = (
                    "rev" if self._client.CLOUD_PLATFORM_SPACES else "revision_id"
                )
                params[rev_param_key] = rev_id

            async with self._client.async_httpx_client.stream(
                method="GET",
                url=artifact_content_url,
                params=params,
                headers=await self._client._aget_headers(),
            ) as response:
                if response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during downloading function.", response
                    )

                downloaded_model = await response.aread()

            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s", artifact_url
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(
                f"Downloading function content with artifact_url: '{artifact_url}' failed.",
                e,
            )

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)

            print(f"Successfully saved function content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving function content with artifact_url: '{filename}' failed.",
                e,
            )

    def delete(
        self, function_id: str | None = None, force: bool = False, **kwargs
    ) -> Literal["SUCCESS"]:
        """Delete a stored function.

        :param function_id: stored function ID
        :type function_id: str

        :param force: if True, the delete operation will proceed even when the function deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client._functions.delete(function_id)
        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        Functions._validate_type(function_id, "function_id", str, True)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        if not force and self._if_deployment_exist_for_asset(function_id):
            raise WMLClientError(
                "Cannot delete function that has existing deployments. Please delete all associated deployments and try again"
            )

        function_endpoint = self._client._href_definitions.get_function_href(
            function_id
        )

        self._logger.debug("Deletion artifact function endpoint: %s", function_endpoint)

        response = requests.delete(
            function_endpoint,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "function deletion", response, False)

    async def adelete(
        self, function_id: str, force: bool = False
    ) -> Literal["SUCCESS"]:
        """Delete a stored function asynchronously.

        :param function_id: stored function ID
        :type function_id: str

        :param force: if True, the delete operation will proceed even when the function deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client._functions.adelete(function_id)
        """
        Functions._validate_type(function_id, "function_id", str, True)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        if not force and await self._aif_deployment_exist_for_asset(function_id):
            raise WMLClientError(
                "Cannot delete function that has existing deployments. Please delete all associated deployments and try again"
            )

        function_endpoint = self._client._href_definitions.get_function_href(
            function_id
        )

        self._logger.debug("Deletion artifact function endpoint: %s", function_endpoint)

        response = await self._client.async_httpx_client.delete(
            function_endpoint,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(204, "function deletion", response, False)

    def get_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Get metadata of function(s). If neither function ID nor function name is specified,
        the metadata of all functions is returned.
        If only function name is specified, metadata of functions with the name is returned (if any).

        :param function_id: ID of the function
        :type: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `function_id` is None
        :type spec_state: SpecStates, optional

        :param function_name: name of the function, can be used only when `function_id` is None
        :type function_name: str, optional

        :return: metadata of the function
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state=True` may break set `limit`,
            returning less records than stated by set `limit`.

        **Examples**

        .. code-block:: python

            function_details = client._functions.get_details(function_id)
            function_details = client._functions.get_details(function_name='Sample_function')
            function_details = client._functions.get_details()
            function_details = client._functions.get_details(limit=100)
            function_details = client._functions.get_details(limit=100, get_all=True)
            function_details = []
            for entry in client._functions.get_details(limit=100, asynchronous=True, get_all=True):
                function_details.extend(entry)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function", True)

        Functions._validate_type(function_id, "function_uid", str, False)
        Functions._validate_type(limit, "limit", int, False)
        Functions._validate_type(asynchronous, "asynchronous", bool, False)
        Functions._validate_type(get_all, "get_all", bool, False)
        Functions._validate_type(spec_state, "spec_state", object, False)

        if limit and spec_state:
            print(
                "Warning: In current implementation setting `spec_state=True` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Functions._validate_type(function_id, "function_id", str, False)
        Functions._validate_type(limit, "limit", int, False)

        url = self._client._href_definitions.get_functions_href()

        if function_id is not None:
            return self._get_artifact_details(url, function_id, limit, "functions")

        if spec_state:
            filter_func = self._get_filter_func_by_spec_ids(
                self._get_and_cache_spec_ids_for_state(spec_state)
            )
        elif function_name:
            filter_func = self._get_filter_func_by_artifact_name(function_name)
        else:
            filter_func = None

        return self._get_artifact_details(
            url,
            function_id,
            limit,
            "functions",
            _async=asynchronous,
            _all=get_all,
            _filter_func=filter_func,
        )

    async def aget_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
    ) -> dict:
        """Get metadata of function(s) asynchronously. If neither function ID nor function name is specified,
        the metadata of all functions is returned.
        If only function name is specified, metadata of functions with the name is returned (if any).

        :param function_id: ID of the function
        :type: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `function_id` is None
        :type spec_state: SpecStates, optional

        :param function_name: name of the function, can be used only when `function_id` is None
        :type function_name: str, optional

        :return: metadata of the function
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state=True` may break set `limit`,
            returning less records than stated by set `limit`.

        **Examples**

        .. code-block:: python

            function_details = await client._functions.aget_details(function_id)
            function_details = await client._functions.aget_details(function_name='Sample_function')
            function_details = await client._functions.aget_details()
            function_details = await client._functions.aget_details(limit=100)
            function_details = await client._functions.aget_details(limit=100, get_all=True)
            function_details = []
            async for entry in await client._functions.aget_details(limit=100, asynchronous=True, get_all=True):
                function_details.extend(entry)

        """

        Functions._validate_type(function_id, "function_uid", str, False)
        Functions._validate_type(limit, "limit", int, False)
        Functions._validate_type(asynchronous, "asynchronous", bool, False)
        Functions._validate_type(get_all, "get_all", bool, False)
        Functions._validate_type(spec_state, "spec_state", object, False)

        if limit and spec_state:
            print(
                "Warning: In current implementation setting `spec_state=True` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Functions._validate_type(function_id, "function_id", str, False)
        Functions._validate_type(limit, "limit", int, False)

        url = self._client._href_definitions.get_functions_href()

        if function_id is not None:
            return await self._aget_artifact_details(
                url, function_id, limit, "functions"
            )

        if spec_state:
            filter_func = self._get_filter_func_by_spec_ids(
                self._get_and_cache_spec_ids_for_state(spec_state)
            )
        elif function_name:
            filter_func = self._get_filter_func_by_artifact_name(function_name)
        else:
            filter_func = None

        return await self._aget_artifact_details(
            url,
            function_id,
            limit,
            "functions",
            _async=asynchronous,
            _all=get_all,
            _filter_func=filter_func,
        )

    @classmethod
    def get_id(cls, function_details: dict) -> str:
        """Get ID of stored function.

        :param function_details: metadata of the stored function
        :type function_details: dict

        :return: ID of stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_id)
            function_id = client._functions.get_id(function_details)
        """

        cls._validate_type(function_details, "function_details", object, True)

        if "asset_id" in function_details["metadata"]:
            return cls._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "asset_id"]
            )

        if "guid" in function_details["metadata"]:
            cls._validate_type_of_details(function_details, FUNCTION_DETAILS_TYPE)
            return cls._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "guid"]
            )

        return cls._get_required_element_from_dict(
            function_details, "function_details", ["metadata", "id"]
        )

    @classmethod
    def get_uid(cls, function_details: dict) -> str:
        """Get UID of stored function.

        *Deprecated:* Use get_id(function_details) instead.

        :param function_details: metadata of the stored function
        :type function_details: dict

        :return: UID of stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_uid)
            function_uid = client._functions.get_uid(function_details)
        """

        get_uid_method_deprecated_warning = (
            "This method is deprecated, please use `get_id(function_details)` instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)

        return cls.get_id(function_details)

    @classmethod
    def get_href(cls, function_details: dict) -> str:
        """Get the URL of a stored function.

        :param function_details: details of the stored function
        :type function_details: dict

        :return: href of the stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_id)
            function_url = client._functions.get_href(function_details)
        """

        cls._validate_type(function_details, "function_details", object, True)

        if "asset_type" in function_details["metadata"]:
            return cls._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "href"]
            )

        if "href" in function_details["metadata"]:
            cls._validate_type_of_details(function_details, FUNCTION_DETAILS_TYPE)
            return cls._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "href"]
            )

        # Cloud Convergence
        return f"/ml/v4/functions/{function_details['metadata']['id']}"

    def list(self, limit: int | None = None) -> pandas.DataFrame:
        """Return stored functions in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed functions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._functions.list()
        """
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        function_resources = self.get_details(
            get_all=self._should_get_all_values(limit)
        )["resources"]

        function_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
                m["entity"]["type"] if "type" in m["entity"] else None,
                self._client.software_specifications._get_state(m),
                self._client.software_specifications._get_replacement(m),
            )
            for m in function_resources
        ]

        return self._list(
            function_values,
            ["ID", "NAME", "CREATED", "TYPE", "SPEC_STATE", "SPEC_REPLACEMENT"],
            limit,
        )

    def clone(
        self,
        function_id: str | None = None,
        space_id: str | None = None,
        action: str = "copy",
        rev_id: str | None = None,
        **kwargs,
    ) -> None:
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        raise WMLClientError(Messages.get_message(message_id="cloning_not_supported"))

    def create_revision(self, function_id: str | None = None, **kwargs) -> dict:
        """Create a new function revision.

        :param function_id: unique ID of the function
        :type function_id: str

        :return: revised metadata of the stored function
        :rtype: dict

        **Example:**

        .. code-block:: python

            client._functions.create_revision(function_id)
        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        Functions._validate_type(function_id, "function_id", str, False)

        return self._create_revision_artifact(
            self._client._href_definitions.get_functions_href(),
            function_id,
            "functions",
        )

    async def acreate_revision(self, function_id: str) -> dict:
        """Create a new function revision asynchronously.

        :param function_id: unique ID of the function
        :type function_id: str

        :return: revised metadata of the stored function
        :rtype: dict

        **Example:**

        .. code-block:: python

            await client._functions.acreate_revision(function_id)
        """

        Functions._validate_type(function_id, "function_id", str, False)

        return await self._acreate_revision_artifact(
            self._client._href_definitions.get_functions_href(),
            function_id,
            "functions",
        )

    def get_revision_details(self, function_id: str, rev_id: str, **kwargs) -> dict:
        """Get metadata of a specific revision of a stored function.

        :param function_id: definition of the stored function
        :type function_id: str

        :param rev_id: unique ID of the function revision
        :type rev_id: str

        :return: stored function revision metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            function_revision_details = client._functions.get_revision_details(function_id, rev_id)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")

        if rev_id is not None and isinstance(rev_id, int):
            rev_id_as_int_deprecated_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_deprecated_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        self._client._check_if_either_is_set()
        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(rev_id, "rev_id", str, True)

        return self._get_with_or_without_limit(
            self._client._href_definitions.get_function_href(function_id),
            limit=None,
            op_name="function",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    async def aget_revision_details(self, function_id: str, rev_id: str) -> dict:
        """Get metadata of a specific revision of a stored function asynchronously.

        :param function_id: definition of the stored function
        :type function_id: str

        :param rev_id: unique ID of the function revision
        :type rev_id: str

        :return: stored function revision metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            function_revision_details = await client._functions.aget_revision_details(function_id, rev_id)

        """

        self._client._check_if_either_is_set()
        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(rev_id, "rev_id", str, True)

        return await self._aget_with_or_without_limit(
            self._client._href_definitions.get_function_href(function_id),
            limit=None,
            op_name="function",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def list_revisions(
        self, function_id: str | None = None, limit: int | None = None, **kwargs
    ) -> pandas.DataFrame:
        """Print all revisions for a given function ID in a table format.

        :param function_id: unique ID of the stored function
        :type function_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._functions.list_revisions(function_id)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        self._client._check_if_either_is_set()

        Functions._validate_type(function_id, "function_id", str, True)

        # CP4D logic is wrong. By passing "revisions" in second param above for _get_artifact_details()
        # it won't even consider limit value and also GUID gives only rev number, not actual guid

        function_details = self._get_artifact_details(
            self._client._href_definitions.get_function_revisions_href(function_id),
            None,
            None,
            "function revisions",
            _all=self._should_get_all_values(limit),
        )

        function_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["rev"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in function_details["resources"]
        ]

        return self._list(
            function_values,
            ["ID", "REV", "NAME", "CREATED"],
            limit,
        )

    @staticmethod
    def _prepare_function_content(
        function: str | Callable,
    ) -> tuple[str, bool, str | None]:
        """Prepare function content for storing in the repository.
        If a Callable is passed, this function:
            - removes unnecessary indentation
            - validates and injects default parameters' values if ``function`` returns scoring function
            - creates an archive if ``function`` is a Callable

        :param function: _description_
        :type function: Union[str, Callable]

        :raises UnexpectedType: if any of ``function`` default parameters is not of basic Python type

        :raises WMLClientError: if ``function`` is defined incorrectly

        :return: path to compressed function source if archive is provided by user, name of the archive if not provided by user
        :rtype: tuple[str, bool, Optional[str]]
        """

        if isinstance(function, str):
            return function, True, None

        user_content_file = False
        archive_name = None

        try:
            code = inspect.getsource(function).split("\n")
            r = re.compile(r"^ *")
            m = r.search(code[0])
            intend = m.group(0)  # type: ignore[union-attr]

            code = [line.replace(intend, "", 1) for line in code]

            args_spec = inspect.getfullargspec(function)

            defaults = args_spec.defaults if args_spec.defaults is not None else []
            args = args_spec.args if args_spec.args is not None else []

            if function.__name__ == "score":
                code = "\n".join(code)
                file_content = cast(str, code)
            elif len(args) == len(defaults):
                for i, d in enumerate(defaults):
                    if not is_of_python_basic_type(d):
                        raise UnexpectedType(args[i], "primitive python type", type(d))

                args_pattern = ",".join([rf"\s*{arg}\s*=\s*(.+)\s*" for arg in args])
                pattern = rf"^def {function.__name__}\s*\({args_pattern}\)\s*:"
                code = "\n".join(code)
                res = re.match(pattern, code)  # type: ignore[call-overload]

                for i in range(len(defaults) - 1, -1, -1):
                    default = defaults[i]
                    code = (
                        code[: res.start(i + 1)]
                        + repr(default)
                        + code[res.end(i + 1) :]
                    )

                code += f"\n\nscore = {function.__name__}()"

                file_content = cast(str, code)
            else:
                raise WMLClientError(
                    "Function passed is not 'score' function nor generator function. Generator function should have no arguments or all arguments with primitive python default values."
                )

            tmp_uid = f"tmp_python_function_code_{str(uuid.uuid4()).replace('-', '_')}"
            filename = f"{tmp_uid}.py"

            with open(filename, "w") as f:
                f.write(file_content)

            archive_name = f"{filename}.gz"

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

            raise WMLClientError("Exception during getting function code.", e)
