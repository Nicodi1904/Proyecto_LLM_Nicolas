#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal

from pandas import DataFrame

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.metanames import PkgExtnMetaNames
from ibm_watsonx_ai.utils import PKG_EXTN_DETAILS_TYPE, content_type_for
from ibm_watsonx_ai.utils.utils import AsyncFileReader
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    ResourceIdByNameNotFound,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class PkgExtn(WMLResource):
    """Store and manage software Packages Extension specs."""

    ConfigurationMetaNames = PkgExtnMetaNames()
    """MetaNames for Package Extensions creation."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def _get_required_element_from_response(self, response_data: dict) -> dict:
        WMLResource._validate_type(response_data, "pkg_extn_response", dict)

        if self._client.default_space_id is not None:
            new_el = {
                "metadata": {
                    "space_id": response_data["metadata"]["space_id"],
                    "name": response_data["metadata"]["name"],
                    "asset_id": response_data["metadata"]["asset_id"],
                    "asset_type": response_data["metadata"]["asset_type"],
                    "created_at": response_data["metadata"]["created_at"],
                },
                "entity": response_data["entity"],
            }
        elif self._client.default_project_id is not None:
            new_el = {
                "metadata": {
                    "project_id": response_data["metadata"]["project_id"],
                    "name": response_data["metadata"]["name"],
                    "asset_id": response_data["metadata"]["asset_id"],
                    "asset_type": response_data["metadata"]["asset_type"],
                    "created_at": response_data["metadata"]["created_at"],
                },
                "entity": response_data["entity"],
            }
        else:
            raise ValueError("WML client should have set default project or space id.")

        if "href" in response_data["metadata"]:
            href_without_host = response_data["metadata"]["href"].split(".com")[-1]
            new_el["metadata"].update({"href": href_without_host})

        return new_el

    def get_details(self, pkg_extn_id: str) -> dict:
        """Get package extensions details.

        :param pkg_extn_id: unique ID of the package extension
        :type pkg_extn_id: str

        :return: details of the package extension
        :rtype: dict

        **Example:**

        .. code-block:: python

            pkg_extn_details = client.pkg_extn.get_details(pkg_extn_id)

        """
        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        response = requests.get(
            self._client._href_definitions.get_pkg_extn_href(pkg_extn_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        response_data = self._handle_response(
            200, "get package extension details", response
        )
        if response.status_code == 200:
            return self._get_required_element_from_response(response_data)
        else:
            return response_data

    async def aget_details(self, pkg_extn_id: str) -> dict:
        """Get package extensions details asynchronously.

        :param pkg_extn_id: unique ID of the package extension
        :type pkg_extn_id: str

        :return: details of the package extension
        :rtype: dict

        **Example:**

        .. code-block:: python

            pkg_extn_details = await client.pkg_extn.aget_details(pkg_extn_id)

        """
        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_pkg_extn_href(pkg_extn_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        response_data = self._handle_response(
            200, "get package extension details", response
        )
        if response.status_code == 200:
            return self._get_required_element_from_response(response_data)
        else:
            return response_data

    def _create_pkg_extn_asset(self, pkg_extn_metadata: dict) -> dict:
        print("Creating package extension")

        pkg_extn_meta_json = json.dumps(pkg_extn_metadata)

        response = requests.post(
            self._client._href_definitions.get_pkg_extns_href(),
            params=self._client._params(),
            headers=self._client._get_headers(),
            data=pkg_extn_meta_json,
        )
        if response.status_code != 201:
            raise WMLClientError(
                "Failed while creating package extension", response.text
            )

        pkg_extn_details = self._handle_response(
            201, "creating new package extension", response
        )

        return pkg_extn_details

    def _upload_pkg_extn_file(self, file_path: str, pkg_extn_details: dict) -> None:
        pkg_extn_asset_id = self.get_id(pkg_extn_details)
        pkg_extn_presigned_url = self.get_href(pkg_extn_details)

        if self._client.ICP_PLATFORM_SPACES:
            pkg_extn_presigned_url = (
                self._client.credentials.url + pkg_extn_presigned_url  # type: ignore
            )

        try:
            if os.stat(file_path).st_size == 0:
                raise WMLClientError("Package extension file cannot be empty")

            with open(file_path, "rb") as file_object:
                if self._client.CLOUD_PLATFORM_SPACES:
                    content_type = content_type_for(filepath=file_path)
                    response = requests.put(
                        pkg_extn_presigned_url,
                        data=file_object,
                        headers={"Content-Type": content_type},
                    )
                else:
                    response = requests.put(
                        pkg_extn_presigned_url,
                        files={
                            "file": (
                                file_path,
                                file_object,
                                "application/octet-stream",
                            )
                        },
                    )
        except Exception as e:
            deletion_response = requests.delete(
                self._client._href_definitions.get_pkg_extn_href(pkg_extn_asset_id),
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            print(deletion_response.status_code)
            raise WMLClientError("Failed while reading a file.", e)

        if response.status_code in (200, 201):
            return

        try:
            self.delete(pkg_extn_asset_id)
        except Exception:
            pass

        raise WMLClientError("Failed while creating package extension", response.text)

    def _mark_upload_as_complete(self, pkg_extn_details: dict) -> dict:
        pkg_extn_asset_id = pkg_extn_details["metadata"]["asset_id"]

        response = requests.post(
            self._client._href_definitions.get_pkg_extn_upload_complete_href(
                pkg_extn_asset_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        if response.status_code == 204:
            print("SUCCESS")
            return self._get_required_element_from_response(pkg_extn_details)

        try:
            self.delete(pkg_extn_asset_id)
        except Exception:
            pass

        raise WMLClientError("Failed while creating package extension", response.text)

    def store(self, meta_props: dict, file_path: str) -> dict:
        """Create a package extension.

        :param meta_props: metadata of the package extension. To see available meta names, use:

            .. code-block:: python

                client.package_extensions.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param file_path: path to the file to be uploaded as a package extension
        :type file_path: str

        :return: metadata of the package extension
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.package_extensions.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
                client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
                client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
            }

            pkg_extn_details = client.package_extensions.store(meta_props=meta_props, file_path="/path/to/file")

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        PkgExtn._validate_type(meta_props, "meta_props", dict, True)
        PkgExtn._validate_type(file_path, "file_path", str, True)

        pkg_extn_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        pkg_extn_details = self._create_pkg_extn_asset(pkg_extn_meta)

        self._upload_pkg_extn_file(file_path, pkg_extn_details)

        return self._mark_upload_as_complete(pkg_extn_details)

    async def _acreate_pkg_extn_asset(self, pkg_extn_metadata: dict) -> dict:
        print("Creating package extension")

        pkg_extn_meta_json = json.dumps(pkg_extn_metadata)

        response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_pkg_extns_href(),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
            content=pkg_extn_meta_json,
        )

        if response.status_code != 201:
            raise WMLClientError(
                "Failed while creating package extension", response.text
            )

        pkg_extn_details = self._handle_response(
            201, "creating new package extension", response
        )

        return pkg_extn_details

    async def _aupload_pkg_extn_file(
        self, file_path: str, pkg_extn_details: dict
    ) -> None:
        pkg_extn_asset_id = self.get_id(pkg_extn_details)
        pkg_extn_presigned_url = self.get_href(pkg_extn_details)

        if self._client.ICP_PLATFORM_SPACES:
            pkg_extn_presigned_url = (
                self._client.credentials.url + pkg_extn_presigned_url  # type: ignore
            )

        try:
            if os.stat(file_path).st_size == 0:
                raise WMLClientError("Package extension file cannot be empty")

            if self._client.CLOUD_PLATFORM_SPACES:
                content_type = content_type_for(filepath=file_path)
                response = await self._client.async_httpx_client.put(
                    pkg_extn_presigned_url,
                    content=AsyncFileReader(file_path),
                    headers={"Content-Type": content_type},
                )
            else:
                with open(file_path, "rb") as file_object:
                    response = await self._client.async_httpx_client.put(
                        pkg_extn_presigned_url,
                        files={
                            "file": (
                                file_path,
                                file_object,
                                "application/octet-stream",
                            )
                        },
                    )
        except Exception as e:
            deletion_response = await self._client.async_httpx_client.delete(
                self._client._href_definitions.get_pkg_extn_href(pkg_extn_asset_id),
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )
            print(deletion_response.status_code)
            raise WMLClientError("Failed while reading a file.", e)

        if response.status_code in (200, 201):
            return

        try:
            await self.adelete(pkg_extn_asset_id)
        except Exception:
            pass

        raise WMLClientError("Failed while creating package extension", response.text)

    async def _amark_upload_as_complete(self, pkg_extn_details: dict) -> dict:
        pkg_extn_asset_id = self.get_id(pkg_extn_details)

        response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_pkg_extn_upload_complete_href(
                pkg_extn_asset_id
            ),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )
        if response.status_code == 204:
            print("SUCCESS")
            return self._get_required_element_from_response(pkg_extn_details)

        try:
            await self.adelete(pkg_extn_asset_id)
        except Exception:
            pass

        raise WMLClientError("Failed while creating package extension", response.text)

    async def astore(self, meta_props: dict, file_path: str) -> dict:
        """Create a package extension asynchronously.

        :param meta_props: metadata of the package extension. To see available meta names, use:

            .. code-block:: python

                client.package_extensions.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param file_path: path to the file to be uploaded as a package extension
        :type file_path: str

        :return: metadata of the package extension
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.package_extensions.ConfigurationMetaNames.NAME: "skl_pipeline_heart_problem_prediction",
                client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "description scikit-learn_0.20",
                client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
            }

            pkg_extn_details = await client.package_extensions.astore(meta_props=meta_props, file_path="/path/to/file")

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        PkgExtn._validate_type(meta_props, "meta_props", dict, True)
        PkgExtn._validate_type(file_path, "file_path", str, True)

        pkg_extn_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        pkg_extn_details = await self._acreate_pkg_extn_asset(pkg_extn_meta)

        await self._aupload_pkg_extn_file(file_path, pkg_extn_details)

        return await self._amark_upload_as_complete(pkg_extn_details)

    def list(self) -> DataFrame:
        """List the package extensions in a table format.

        :return: pandas.DataFrame with listed package extensions
        :rtype: pandas.DataFrame

        .. code-block:: python

            client.package_extensions.list()

        """

        response = requests.get(
            self._client._href_definitions.get_pkg_extns_href(),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        self._handle_response(200, "list pkg_extn", response)
        asset_details = self._handle_response(200, "list assets", response)["resources"]
        pkg_extn_values = [
            (
                m["metadata"]["name"],
                m["metadata"]["asset_id"],
                m["entity"]["package_extension"]["type"],
                m["metadata"]["created_at"],
            )
            for m in asset_details
        ]

        table = self._list(
            pkg_extn_values,
            ["NAME", "ASSET_ID", "TYPE", "CREATED_AT"],
            None,
        )
        return table

    @staticmethod
    def get_id(pkg_extn_details: dict) -> str:
        """Get the unique ID of a package extension.

        :param pkg_extn_details: details of the package extension
        :type pkg_extn_details: dict

        :return: unique ID of the package extension
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.package_extensions.get_id(pkg_extn_details)

        """
        PkgExtn._validate_type(pkg_extn_details, "pkg_extn_details", object, True)
        PkgExtn._validate_type_of_details(pkg_extn_details, PKG_EXTN_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            pkg_extn_details, "pkg_extn_details", ["metadata", "asset_id"]
        )

    def get_id_by_name(self, pkg_extn_name: str) -> str:
        """Get the ID of a package extension.

        :param pkg_extn_name: name of the package extension
        :type pkg_extn_name: str

        :return: unique ID of the package extension
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.package_extensions.get_id_by_name(pkg_extn_name)

        """
        PkgExtn._validate_type(pkg_extn_name, "pkg_extn_name", str, True)

        parameters = self._client._params()
        parameters.update(name=pkg_extn_name)

        response = requests.get(
            self._client._href_definitions.get_pkg_extns_href(),
            params=parameters,
            headers=self._client._get_headers(),
        )

        total_values = self._handle_response(200, "get pkg extn", response)[
            "total_results"
        ]
        if total_values != 0:
            pkg_extn_details = self._handle_response(200, "get pkg extn", response)[
                "resources"
            ]
            return pkg_extn_details[0]["metadata"]["asset_id"]
        else:
            raise ResourceIdByNameNotFound(pkg_extn_name, "package extension")

    async def aget_id_by_name(self, pkg_extn_name: str) -> str:
        """Get the ID of a package extension asynchronously.

        :param pkg_extn_name: name of the package extension
        :type pkg_extn_name: str

        :return: unique ID of the package extension
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = await client.package_extensions.aget_id_by_name(pkg_extn_name)

        """
        PkgExtn._validate_type(pkg_extn_name, "pkg_extn_name", str, True)

        parameters = self._client._params()
        parameters["name"] = pkg_extn_name

        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_pkg_extns_href(),
            params=parameters,
            headers=await self._client._aget_headers(),
        )

        response_details = self._handle_response(200, "get pkg extn", response)
        if not response_details["total_results"]:
            raise ResourceIdByNameNotFound(pkg_extn_name, "package extension")

        return response_details["resources"][0]["metadata"]["asset_id"]

    @staticmethod
    def get_href(pkg_extn_details: dict) -> str:
        """Get the URL of a stored package extension.

        :param pkg_extn_details: details of the package extension
        :type pkg_extn_details: dict

        :return: href of the package extension
        :rtype: str

        **Example:**

        .. code-block:: python

            pkg_extn_details = client.package_extensions.get_details(pkg_extn_id)
            pkg_extn_href = client.package_extensions.get_href(pkg_extn_details)

        """
        PkgExtn._validate_type(pkg_extn_details, "pkg_extn_details", object, True)
        PkgExtn._validate_type_of_details(pkg_extn_details, PKG_EXTN_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            pkg_extn_details,
            "pkg_extn_details",
            ["entity", "package_extension", "href"],
        )

    def delete(self, pkg_extn_id: str) -> Literal["SUCCESS"]:
        """Delete a package extension.

        :param pkg_extn_id: unique ID of the package extension
        :type pkg_extn_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client.package_extensions.delete(pkg_extn_id)

        """
        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        response = requests.delete(
            self._client._href_definitions.get_pkg_extn_href(pkg_extn_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete pkg extn specification", response)

    async def adelete(self, pkg_extn_id: str) -> Literal["SUCCESS"]:
        """Delete a package extension asynchronously.

        :param pkg_extn_id: unique ID of the package extension
        :type pkg_extn_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client.package_extensions.adelete(pkg_extn_id)

        """
        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        response = await self._client.async_httpx_client.delete(
            self._client._href_definitions.get_pkg_extn_href(pkg_extn_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(204, "delete pkg extn specification", response)

    def download(self, pkg_extn_id: str, filename: str) -> str:
        """Download a package extension.

        :param pkg_extn_id: unique ID of the package extension to be downloaded
        :type pkg_extn_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :return: path to the downloaded package extension content
        :rtype: str

        **Example:**

        .. code-block:: python

            client.package_extensions.download(pkg_extn_id,"sample_conda.yml/custom_library.zip")

        """

        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        pkg_extn_details = self.get_details(pkg_extn_id)

        artifact_content_url = self.get_href(pkg_extn_details)
        if self._client.ICP_PLATFORM_SPACES:
            artifact_content_url = self._credentials.url + artifact_content_url  # type: ignore

        response = requests.get(artifact_content_url)
        if response.status_code != 200:
            raise ApiRequestFailure(
                "Failure during downloading package extension.",
                response,
            )

        downloaded_asset = response.content
        try:
            with open(filename, "wb") as f:
                f.write(downloaded_asset)
        except IOError as e:
            raise WMLClientError(
                f"Saving asset with artifact_url: '{filename}' failed.",
                e,
            )

        print(f"Successfully saved package extension content to file: '{filename}'")
        return os.path.abspath(filename)

    async def adownload(self, pkg_extn_id: str, filename: str) -> str:
        """Download a package extension asynchronously.

        :param pkg_extn_id: unique ID of the package extension to be downloaded
        :type pkg_extn_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :return: path to the downloaded package extension content
        :rtype: str

        **Example:**

        .. code-block:: python

            file_path = await client.package_extensions.adownload(
                pkg_extn_id, "sample_conda.yml/custom_library.zip"
            )

        """

        PkgExtn._validate_type(pkg_extn_id, "pkg_extn_id", str, True)

        pkg_extn_details = await self.aget_details(pkg_extn_id)

        artifact_content_url = self.get_href(pkg_extn_details)
        if self._client.ICP_PLATFORM_SPACES:
            artifact_content_url = self._credentials.url + artifact_content_url  # type: ignore

        response = await self._client.async_httpx_client.get(artifact_content_url)
        if response.status_code != 200:
            raise ApiRequestFailure(
                "Failure during {}.".format("downloading package extension"),
                response,
            )

        downloaded_asset = response.content
        try:
            with open(filename, "wb") as file:
                file.write(downloaded_asset)
        except IOError as e:
            raise WMLClientError(
                f"Saving asset with artifact_url: '{artifact_content_url}' failed.",
                e,
            )

        print(f"Successfully saved package extension content to file: '{filename}'")
        return os.path.abspath(filename)
