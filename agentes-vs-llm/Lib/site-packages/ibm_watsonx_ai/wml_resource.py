#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Iterable,
    Literal,
    TypeAlias,
    cast,
    overload,
)

import httpx

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.utils import (
    get_type_of_details,
    next_resource_generator,
)
from ibm_watsonx_ai.utils.utils import (
    _get_id_from_deprecated_uid,
    anext_resource_generator,
    get_from_json,
)
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    MissingMetaProp,
    MissingValue,
    NoWMLCredentialsProvided,
    UnexpectedType,
    WMLClientError,
)

if TYPE_CHECKING:
    import pandas as pd
    import requests as _requests

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.sw_spec import SpecStates

    ArtifactDetailsType: TypeAlias = Generator | dict[str, Any]
    AsyncArtifactDetailsType: TypeAlias = AsyncGenerator | dict[str, Any]
    ResponseLike: TypeAlias = _requests.Response | httpx.Response

_DEFAULT_LIST_BATCH_SIZE_LIMIT = 200

httpx_handled = [
    "ibm_watsonx_ai.deployments",
    "ibm_watsonx_ai.foundation_models.ilab.taxonomies",
    "ibm_watsonx_ai.foundation_models.ilab.documents",
    "ibm_watsonx_ai.foundation_models.ilab.synthetic_data",
    "ibm_watsonx_ai.foundation_models_manager",
    "ibm_watsonx_ai.foundation_models.inference.fm_model_inference",
]


class WMLResource:
    _logger = logging.getLogger(__name__)

    def __init__(self, name: str, client: APIClient):
        self._name = name
        WMLResource._validate_type(client, "client", object, True)
        if client.credentials is None:
            raise NoWMLCredentialsProvided
        WMLResource._validate_type(client.credentials, "credentials", Credentials, True)
        self._client = client

    @property
    def _credentials(self):
        return self._client.credentials

    @overload
    @classmethod
    def _handle_response(
        cls,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: Literal[True] = True,
        _silent_response_logging: bool = False,
        _field_to_hide: str | None = None,
    ) -> dict: ...

    @overload
    @classmethod
    def _handle_response(
        cls,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: Literal[False],
        _silent_response_logging: bool = ...,
        _field_to_hide: str | None = ...,
    ) -> str: ...

    @classmethod
    def _handle_response(
        cls,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: bool = True,
        _silent_response_logging: bool = False,
        _field_to_hide: str | None = None,
    ) -> dict | str:
        """Internal method for handling HTTP requests responses.

        :param _silent_response_logging: If True the whole response text is not visible in logging messages, defaults to False
        :type _silent_response_logging: bool, optional

        :param _field_to_hide: Determine what field in the response should be hide in logging, defaults to None
        :type _field_to_hide: str | None, optional
        """
        if "dele" in operationName or "cancel" in operationName:
            if response.status_code == expected_status_code:
                return "SUCCESS"
            else:
                msg = f"{operationName} failed. Reason: {response.text}"
                raise WMLClientError(msg)

        if response.status_code == expected_status_code:
            cls._logger.info(
                "Successfully finished %s for url: '%s'",
                operationName,
                response.url,
            )

            if cls._logger.level <= logging.DEBUG:
                if _field_to_hide is not None:
                    replace_value = "..."
                    try:

                        def decode_dict(processed_dict: dict) -> dict:
                            if _field_to_hide in processed_dict.keys():
                                processed_dict[_field_to_hide] = replace_value
                            return processed_dict

                        response_text = json.dumps(
                            response.json(object_hook=decode_dict)
                        )
                    except Exception:
                        response_text = response.text
                else:
                    response_text = response.text

                cls._logger.debug(
                    "Response(%s %s %s)%s",
                    response.request.method,
                    response.url,
                    response.status_code,
                    (f": {response_text}" if not _silent_response_logging else ""),
                )
            if json_response:
                try:
                    return response.json()
                except Exception as e:
                    raise WMLClientError(
                        f"Failure during parsing json response: '{response.text}'",
                        e,
                    )
            else:
                return response.text
        else:
            raise ApiRequestFailure(
                f"Failure during {operationName}.",
                response,
            )

    @staticmethod
    def _get_required_element_from_dict(
        el: dict, root_path: str, path: list[str]
    ) -> str | dict:
        WMLResource._validate_type(el, root_path, dict)
        WMLResource._validate_type(root_path, "root_path", str)
        WMLResource._validate_type(path, "path", list)

        if len(path) < 1:
            raise WMLClientError(f"Unexpected path length: {len(path)}")

        try:
            new_el = el[path[0]]
            new_path = path[1:]
        except Exception as e:
            raise MissingValue(root_path + "." + str(path[0]), e)

        if len(path) > 1:
            return WMLResource._get_required_element_from_dict(
                new_el, root_path + "." + path[0], new_path
            )
        else:
            if new_el is None:
                raise MissingValue(root_path + "." + str(path[0]))

            return new_el

    def _get_asset_based_resource(
        self,
        asset_id: str,
        asset_type: str,
        get_required_element_from_response: Callable,
        limit: int | None = None,
        get_all: bool | None = None,
    ) -> dict:
        WMLResource._validate_type(asset_id, f"{asset_type}_id", str, False)

        if asset_id:
            response = requests.get(
                self._client._href_definitions.get_asset_href(asset_id),
                params=self._client._params(),
                headers=self._client._get_headers(),
            )

            return get_required_element_from_response(
                self._handle_response(200, f"get {asset_type} details", response)
            )

        href = self._client._href_definitions.get_asset_search_href(asset_type)

        def get_chunk(data):
            response = requests.post(
                href,
                json=data,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )

            return [
                get_required_element_from_response(x)
                for x in self._handle_response(
                    200, f"get {asset_type}s details", response
                )[
                    "results"  # type: ignore[index]
                ]
            ], response.json().get("next")

        data = {"query": "*:*"}
        if asset_type == "data_asset":
            data["include"] = (
                "entity,attachments"  # "Attachments can only be included if entity is also requested."
            )

        result, data = get_chunk(data)

        if get_all:
            while data is not None and (limit is None or len(result) < limit):
                res, data = get_chunk(data)
                result.extend(res)

        return {"resources": result if limit is None else result[:limit]}

    async def _aget_asset_based_resource(
        self,
        asset_id: str | None,
        asset_type: str,
        get_required_element_from_response: Callable,
        limit: int | None = None,
        get_all: bool | None = None,
    ) -> dict:
        WMLResource._validate_type(asset_id, f"{asset_type}_id", str, False)

        if asset_id:
            response = await self._client.async_httpx_client.get(
                self._client._href_definitions.get_asset_href(asset_id),
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )

            return get_required_element_from_response(
                self._handle_response(200, f"get {asset_type} details", response)
            )

        href = self._client._href_definitions.get_asset_search_href(asset_type)

        async def get_chunk(data):
            response = await self._client.async_httpx_client.post(
                href,
                json=data,
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )

            return [
                get_required_element_from_response(x)
                for x in self._handle_response(
                    200, f"get {asset_type}s details", response
                )[
                    "results"  # type: ignore[index]
                ]
            ], response.json().get("next")

        data = {"query": "*:*"}

        result, data = await get_chunk(data)

        if get_all:
            while data is not None and (limit is None or len(result) < limit):
                res, data = await get_chunk(data)
                result.extend(res)

        return {"resources": result if limit is None else result[:limit]}

    def _list_asset_based_resource(
        self, url: str, column_names: list[str], limit: int | None = None
    ) -> pd.DataFrame:
        """Lists stored assets in a table format."""

        self._validate_type(url, "url", str, True)
        self._validate_type(column_names, "column_names", list, True)
        self._validate_type(limit, "limit", int, False)

        data: dict[str, Any] = {"query": "*:*"}
        if limit is not None:
            data["limit"] = limit

        response = requests.post(
            url,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=data,
        )
        self._handle_response(200, "list assets", response)
        asset_details = self._handle_response(200, "list assets", response)["results"]
        space_values = [
            [m["metadata"][col.lower()] for col in column_names] for m in asset_details
        ]

        table = self._list(
            space_values,
            column_names,
            limit,
        )
        return table

    async def _alist_asset_based_resource(
        self, url: str, column_names: list[str], limit: int | None = None
    ) -> pd.DataFrame:
        """Lists stored assets in a table format."""

        self._validate_type(url, "url", str, True)
        self._validate_type(column_names, "column_names", list, True)
        self._validate_type(limit, "limit", int, False)

        data: dict[str, Any] = {"query": "*:*"}
        if limit is not None:
            data["limit"] = limit

        response = await self._client.async_httpx_client.post(
            url,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
            json=data,
        )

        self._handle_response(200, "list assets", response)
        asset_details = self._handle_response(200, "list assets", response)["results"]
        space_values = [
            [m["metadata"][col.lower()] for col in column_names] for m in asset_details
        ]

        table = self._list(
            space_values,
            column_names,
            limit,
        )
        return table

    def _delete_asset_based_resource(
        self,
        asset_id: str,
        get_required_element_from_response: Callable,
        purge_on_delete: bool | None = None,
        **kwargs: Any,
    ) -> dict | str:
        """Soft delete the stored asset. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.
        """

        asset_id = _get_id_from_deprecated_uid(kwargs, asset_id, "asset")

        self._validate_type(asset_id, "asset_id", str, True)

        params = self._client._params()

        if purge_on_delete:
            params["purge_on_delete"] = True

        response = requests.delete(
            self._client._href_definitions.get_asset_href(asset_id),
            params=params,
            headers=self._client._get_headers(),
        )
        if response.status_code == 200:
            return get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, "delete assets", response)

    async def _adelete_asset_based_resource(
        self,
        asset_id: str,
        get_required_element_from_response: Callable,
        purge_on_delete: bool | None = None,
    ) -> dict | str:
        """Soft delete the stored asset. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.
        """
        self._validate_type(asset_id, "asset_id", str, True)

        params = self._client._params()

        if purge_on_delete:
            params["purge_on_delete"] = True

        response = await self._client.async_httpx_client.delete(
            self._client._href_definitions.get_asset_href(asset_id),
            params=params,
            headers=await self._client._aget_headers(),
        )

        if response.status_code == 200:
            return get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, "delete assets", response)

    @staticmethod
    def _validate_type(
        el: Any,
        el_name: str,
        expected_type: type | list[type],
        mandatory: bool = True,
        raise_error_for_list: bool = False,
    ) -> bool | None:
        if el_name is None:
            raise MissingValue("el_name")

        if type(el_name) is not str:
            raise UnexpectedType("el_name", str, type(el_name))

        if expected_type is None:
            raise MissingValue("expected_type")

        if type(expected_type) is not type and type(expected_type) is not list:
            raise UnexpectedType("expected_type", "type or list", type(expected_type))

        if type(mandatory) is not bool:
            raise UnexpectedType("mandatory", bool, type(mandatory))

        if mandatory and el is None:
            raise MissingValue(el_name)
        elif el is None:
            return None

        if type(expected_type) is list:
            try:
                next((x for x in expected_type if isinstance(el, x)))
                return True
            except StopIteration:
                if raise_error_for_list:
                    raise UnexpectedType(el_name, expected_type, type(el))
                return False  # keep for backward compatibilty
        else:
            if not isinstance(el, expected_type):
                raise UnexpectedType(el_name, expected_type, type(el))
        return None

    @staticmethod
    def _validate_meta_prop(
        meta_props: dict, name: str, expected_type: type, mandatory: bool = True
    ) -> None:
        if name in meta_props:
            WMLResource._validate_type(
                meta_props[name], "meta_props." + name, expected_type, mandatory
            )
        else:
            if mandatory:
                raise MissingMetaProp(name)

    @staticmethod
    def _validate_type_of_details(details: dict, expected_type: str | list) -> None:
        actual_type = get_type_of_details(details)

        if type(expected_type) is list:
            expected_types = expected_type
        else:
            expected_types = [expected_type]

        if not any([actual_type == exp_type for exp_type in expected_types]):
            logger = logging.getLogger(__name__)
            logger.debug(
                "Unexpected type of '%s', expected: '%s', actual: '%s', occurred for details: %s",
                "details",
                expected_type,
                actual_type,
                details,
            )
            raise UnexpectedType("details", expected_type, actual_type)

    @overload
    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> dict: ...

    @overload
    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> Generator: ...

    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> ArtifactDetailsType:
        op_name = f"getting {resource_name} details"

        if id is None:
            return self._get_with_or_without_limit(
                url=base_url,
                limit=limit,
                op_name=op_name,
                summary=summary,
                pre_defined=pre_defined,
                query_params=query_params,
                _async=_async,
                _all=_all,
                _filter_func=_filter_func,
            )

        if query_params is None:
            params = self._client._params()
        else:
            params = query_params

        if "userfs" in params:
            params.pop("userfs")

        url = base_url + "/" + id

        request_handler = (
            self._client.httpx_client if self._name in httpx_handled else requests
        )

        response_get = request_handler.get(
            url=url,
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, op_name, response_get)

    @overload
    async def _aget_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> dict: ...

    @overload
    async def _aget_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> AsyncGenerator: ...

    async def _aget_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> AsyncArtifactDetailsType:
        op_name = f"getting {resource_name} details"

        if id is None:
            return await self._aget_with_or_without_limit(
                url=base_url,
                limit=limit,
                op_name=op_name,
                summary=summary,
                pre_defined=pre_defined,
                query_params=query_params,
                _async=_async,
                _all=_all,
                _filter_func=_filter_func,
            )

        if query_params is None:
            params = self._client._params()
        else:
            params = query_params

        params.pop("userfs", None)

        url = base_url + "/" + id

        response_get = await self._client.async_httpx_client.get(
            url=url,
            params=params,
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(200, op_name, response_get)

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> dict: ...

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> Generator: ...

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> ArtifactDetailsType: ...

    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> ArtifactDetailsType:
        params = self._client._params(skip_space_project_chk)

        if self._name in httpx_handled:
            request_handler = self._client.httpx_client
            use_httpx = True
        else:
            request_handler = requests
            use_httpx = False

        if query_params is not None:
            params.update(query_params)

        if summary is False:
            params["summary"] = "false"

        if pre_defined is True:
            params["system_runtimes"] = "true"

        if "userfs" in params:
            params.pop("userfs")

        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            params["limit"] = limit
        else:
            params["limit"] = 200

        if revision is not None:
            if "asset_revision" in op_name:
                # CAMS assets api takes 'revision_id' query parameter
                params["revision_id"] = revision
            else:
                params["rev"] = revision

        resources = []

        href = "/".join(url.split("/")[3:])
        url_2 = "/".join(url.split("/")[:3])

        resource_generator = next_resource_generator(
            self._client,
            url_2,
            href,
            params,
            _all,
            _filter_func,
            use_httpx,
            _silent_response_logging=_silent_response_logging,
        )

        if _async:
            return resource_generator

        if _all:
            for entry in resource_generator:
                resources.extend(entry["resources"])

            return {"resources": resources}

        response_get = request_handler.get(
            url=url,
            headers=self._client._get_headers(),
            params=params,
        )

        result = cast(
            dict,
            self._handle_response(
                200,
                op_name,
                response_get,
                _silent_response_logging=_silent_response_logging,
            ),
        )

        if "resources" in result:
            resources.extend(result["resources"])
        elif "metadata" in result:
            resources.append(result)
        else:
            resources.extend(cast(Iterable, result.get("results")))

        return {"resources": _filter_func(resources) if _filter_func else resources}

    @overload
    async def _aget_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> dict: ...

    @overload
    async def _aget_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> AsyncGenerator: ...

    @overload
    async def _aget_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> AsyncArtifactDetailsType: ...

    async def _aget_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
        _silent_response_logging: bool = False,
    ) -> AsyncArtifactDetailsType:
        params = self._client._params(skip_space_project_chk)

        if query_params is not None:
            params.update(query_params)

        if summary is False:
            params["summary"] = "false"

        if pre_defined is True:
            params["system_runtimes"] = "true"

        if "userfs" in params:
            params.pop("userfs")

        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            params["limit"] = limit
        else:
            params["limit"] = 200

        if revision is not None:
            if "asset_revision" in op_name:
                # CAMS assets api takes 'revision_id' query parameter
                params["revision_id"] = revision
            else:
                params["rev"] = revision

        resources = []

        href = "/".join(url.split("/")[3:])
        url_2 = "/".join(url.split("/")[:3])

        resource_generator = anext_resource_generator(
            self._client,
            url_2,
            href,
            params,
            _all,
            _filter_func,
            _silent_response_logging=_silent_response_logging,
        )

        if _async:
            return resource_generator

        if _all:
            async for entry in resource_generator:
                resources.extend(entry["resources"])

            return {"resources": resources}

        response_get = await self._client.async_httpx_client.get(
            url=url,
            headers=await self._client._aget_headers(),
            params=params,
        )

        result = cast(
            dict,
            self._handle_response(
                200,
                op_name,
                response_get,
                _silent_response_logging=_silent_response_logging,
            ),
        )

        if "resources" in result:
            resources.extend(result["resources"])
        elif "metadata" in result:
            resources.append(result)
        else:
            resources.extend(cast(Iterable, result.get("results")))

        return {"resources": _filter_func(resources) if _filter_func else resources}

    def _if_deployment_exist_for_asset(self, asset_id: str) -> bool:
        response = requests.get(
            self._client._href_definitions.get_deployments_href(),
            params=self._client._params() | {"asset_id": asset_id},
            headers=self._client._get_headers(),
        )

        response_data = self._handle_response(200, "Get deployment details", response)

        return bool(response_data.get("resources"))

    async def _aif_deployment_exist_for_asset(self, asset_id: str) -> bool:
        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_deployments_href(),
            params=self._client._params() | {"asset_id": asset_id},
            headers=await self._client._aget_headers(),
        )

        response_data = self._handle_response(200, "Get deployment details", response)

        return bool(response_data.get("resources"))

    def _list(
        self,
        values: list,
        header: list,
        limit: int | None,
        sort_by: str | None = "CREATED",
    ) -> pd.DataFrame:
        if sort_by is not None and sort_by in header:
            column_no = header.index(sort_by)
            values = sorted(values, key=lambda x: x[column_no], reverse=True)

        import pandas as pd

        if limit is None:
            return pd.DataFrame(values, columns=header)

        else:
            return pd.DataFrame(values[:limit], columns=header)

    def _create_revision_artifact(
        self, base_url: str, id: str, resource_name: str
    ) -> dict:
        self._client._check_if_either_is_set()

        op_name = f"Creation revision for {resource_name}"
        if self._client.default_project_id is not None:
            input_json = {"project_id": self._client.default_project_id}
        else:
            input_json = {"space_id": self._client.default_space_id}

        url = base_url + "/" + id + "/revisions"
        if self._client.CLOUD_PLATFORM_SPACES:
            params = self._client._params(skip_for_create=True)
            response = requests.post(
                url, headers=self._client._get_headers(), params=params, json=input_json
            )
        else:  # ICP_PLATFORM_SPACES
            response = requests.post(
                url,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True),
                json=input_json,
            )

        return self._handle_response(201, op_name, response)

    async def _acreate_revision_artifact(
        self, base_url: str, id: str, resource_name: str
    ) -> dict:
        self._client._check_if_either_is_set()

        op_name = f"Creation revision for {resource_name}"
        if self._client.default_project_id is not None:
            input_json = {"project_id": self._client.default_project_id}
        else:
            input_json = {"space_id": self._client.default_space_id}

        url = base_url + "/" + id + "/revisions"

        if self._client.CLOUD_PLATFORM_SPACES:
            params = self._client._params(skip_for_create=True)
            response = await self._client.async_httpx_client.post(
                url,
                headers=await self._client._aget_headers(),
                params=params,
                json=input_json,
            )
        else:  # ICP_PLATFORM_SPACES
            response = await self._client.async_httpx_client.post(
                url,
                headers=await self._client._aget_headers(),
                params=self._client._params(skip_for_create=True),
                json=input_json,
            )

        return self._handle_response(201, op_name, response)

    def _create_revision_artifact_for_assets(
        self, id: str, resource_name: str
    ) -> dict | str:
        op_name = f"Creation revision for {resource_name}"

        url = self._client._href_definitions.get_asset_href(id) + "/revisions"
        commit_message = "Revision creation for " + resource_name + " " + id

        payload_json = {"commit_message": commit_message}

        # CAMS revision creation api takes space_id as a query parameter. Hence
        # params has to be passed

        response = requests.post(
            url,
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=payload_json,
        )

        return self._handle_response(201, op_name, response)

    async def _acreate_revision_artifact_for_assets(
        self, id: str, resource_name: str
    ) -> dict | str:
        op_name = f"Creation revision for {resource_name}"

        url = self._client._href_definitions.get_asset_href(id) + "/revisions"
        commit_message = "Revision creation for " + resource_name + " " + id

        payload_json = {"commit_message": commit_message}

        # CAMS revision creation api takes space_id as a query parameter. Hence
        # params has to be passed

        response = await self._client.async_httpx_client.post(
            url,
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=payload_json,
        )

        return self._handle_response(201, op_name, response)

    def _update_attachment_for_assets(
        self,
        asset_type: str,
        asset_id: str,
        file_path: str,
        current_attachment_id: str | None = None,
    ) -> Literal[
        "error_in_marking_attachment_complete",
        "error_in_uploading_attachment",
        "error_in_getting_signed_url",
        "error_in_deleting_existing_attachment",
        "success",
    ]:
        if current_attachment_id is not None:
            # Delete existing attachment to upload new attachment
            attachments_id_url = self._client._href_definitions.get_attachment_href(
                asset_id, current_attachment_id
            )

            delete_attachment_response = requests.delete(
                attachments_id_url,
                headers=self._client._get_headers(),
                params=self._client._params(),
            )

        if (
            delete_attachment_response.status_code != 204
            and current_attachment_id is not None
        ):
            self._logger.error(
                "Error in deleting existing attachment %s for asset %s",
                current_attachment_id,
                asset_id,
            )
            return "error_in_deleting_existing_attachment"

        attachment_meta = {
            "asset_type": asset_type,
            "name": "attachment_" + asset_id,
        }

        attachments_url = self._client._href_definitions.get_attachments_href(asset_id)

        # STEP 3b.
        # Get the signed url from CAMS to upload the attachment
        attachment_response = requests.post(
            attachments_url,
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        attachment_details = cast(
            dict,
            self._handle_response(201, "creating new attachment", attachment_response),
        )
        if attachment_response.status_code != 201:
            self._logger.error(
                "Error in getting signed url for attachment for asset %s", asset_id
            )
            return "error_in_getting_signed_url"

        attachment_id = attachment_details["attachment_id"]
        attachment_url = attachment_details["url1"]

        # STEP 3c.
        # Upload attachment
        with open(file_path, "rb") as f:
            if not self._client.ICP_PLATFORM_SPACES:
                put_response = requests.put(attachment_url, data=f.read())
            else:
                put_response = requests.put(
                    self._credentials.url + attachment_url,
                    files={
                        "file": (
                            attachment_meta["name"],
                            f,
                            "application/octet-stream",
                        )
                    },
                )

        if put_response.status_code not in (200, 201):
            self._logger.error("Error in uploading attachment for asset %s", asset_id)
            return "error_in_uploading_attachment"

        # STEP 3d.
        # Mark attachment complete
        complete_response = requests.post(
            self._client._href_definitions.get_attachment_complete_href(
                asset_id, attachment_id
            ),
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        if complete_response.status_code != 200:
            self._logger.error(
                "Error in marking attachment complete for asset %s", asset_id
            )
            return "error_in_marking_attachment_complete"

        return "success"

    async def _aupdate_attachment_for_assets(
        self,
        asset_type: str,
        asset_id: str,
        file_path: str,
        current_attachment_id: str | None = None,
    ) -> Literal[
        "error_in_marking_attachment_complete",
        "error_in_uploading_attachment",
        "error_in_getting_signed_url",
        "error_in_deleting_existing_attachment",
        "success",
    ]:
        if current_attachment_id is not None:
            # Delete existing attachment to upload new attachment
            attachments_id_url = self._client._href_definitions.get_attachment_href(
                asset_id, current_attachment_id
            )

            delete_attachment_response = await self._client.async_httpx_client.delete(
                attachments_id_url,
                headers=await self._client._aget_headers(),
                params=self._client._params(),
            )

        if (
            delete_attachment_response.status_code != 204
            and current_attachment_id is not None
        ):
            self._logger.error(
                "Error in deleting existing attachment %s for asset %s",
                current_attachment_id,
                asset_id,
            )
            return "error_in_deleting_existing_attachment"

        attachment_meta = {
            "asset_type": asset_type,
            "name": "attachment_" + asset_id,
        }

        attachments_url = self._client._href_definitions.get_attachments_href(asset_id)

        # STEP 3b.
        # Get the signed url from CAMS to upload the attachment
        attachment_response = await self._client.async_httpx_client.post(
            attachments_url,
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        attachment_details = cast(
            dict,
            self._handle_response(201, "creating new attachment", attachment_response),
        )
        if attachment_response.status_code != 201:
            self._logger.error(
                "Error in getting signed url for attachment for asset %s", asset_id
            )
            return "error_in_getting_signed_url"

        attachment_id = attachment_details["attachment_id"]
        attachment_url = attachment_details["url1"]

        # STEP 3c.
        # Upload attachment
        with open(file_path, "rb") as f:
            if not self._client.ICP_PLATFORM_SPACES:
                put_response = await self._client.async_httpx_client.put(
                    attachment_url, data=f.read()
                )
            else:
                put_response = await self._client.async_httpx_client.put(
                    self._credentials.url + attachment_url,
                    files={
                        "file": (
                            attachment_meta["name"],
                            f,
                            "application/octet-stream",
                        )
                    },
                )

        if put_response.status_code not in (200, 201):
            self._logger.error("Error in uploading attachment for asset %s", asset_id)
            return "error_in_uploading_attachment"

        # STEP 3d.
        # Mark attachment complete
        complete_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_attachment_complete_href(
                asset_id, attachment_id
            ),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
        )

        if complete_response.status_code != 200:
            self._logger.error(
                "Error in marking attachment complete for asset %s", asset_id
            )
            return "error_in_marking_attachment_complete"

        return "success"

    def _get_and_cache_spec_ids_for_state(self, spec_state: SpecStates) -> list:
        from ibm_watsonx_ai.sw_spec import SpecStates

        if isinstance(spec_state, str):
            spec_state = SpecStates(spec_state)

        if spec_state not in self._client._spec_ids_per_state:
            sw_spec_details = self._client.software_specifications.get_details(
                state_info=True
            )

            for spec in sw_spec_details["resources"]:
                try:
                    state = SpecStates(
                        self._client.software_specifications._get_spec_state(spec)
                    )

                    if state not in self._client._spec_ids_per_state:
                        self._client._spec_ids_per_state[state] = []

                    if (
                        sw_spec_id := spec["metadata"]["asset_id"]
                    ) not in self._client._spec_ids_per_state[state]:
                        self._client._spec_ids_per_state[state].append(sw_spec_id)
                except Exception:
                    # The values are requested by SpecStates, so even if new state will appear
                    # user will not request that new state. If user will wish to request new state, they will open issue.
                    pass

        return self._client._spec_ids_per_state.get(spec_state, [])

    @staticmethod
    def _should_get_all_values(limit: int | None):
        return limit is None or limit > _DEFAULT_LIST_BATCH_SIZE_LIMIT

    @staticmethod
    def _get_filter_func_by_spec_ids(spec_ids: list) -> Callable:
        def filter_func(resources: list) -> list:
            return [
                r
                for r in resources
                if get_from_json(r, ["entity", "software_spec", "id"]) in spec_ids
            ]

        return filter_func

    @staticmethod
    def _get_filter_func_by_artifact_name(artifact_name: str) -> Callable:
        def filter_func(resources: list) -> list:
            return [
                r
                for r in resources
                if get_from_json(r, ["metadata", "name"]) == artifact_name
            ]

        return filter_func
