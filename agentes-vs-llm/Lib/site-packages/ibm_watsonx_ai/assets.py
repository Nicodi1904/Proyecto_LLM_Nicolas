#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Any
from warnings import warn

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import AssetsMetaNames
from ibm_watsonx_ai.utils import DATA_ASSETS_DETAILS_TYPE
from ibm_watsonx_ai.utils.enums import AssetDuplicateAction
from ibm_watsonx_ai.utils.utils import AsyncFileReader, _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class Assets(WMLResource):
    """Store and manage data assets."""

    ConfigurationMetaNames = AssetsMetaNames()
    """MetaNames for Data Assets creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        asset_id: str | None = None,
        get_all: bool | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get data asset details. If no asset_id is passed, details for all assets are returned.

        :param asset_id: unique ID of the asset
        :type asset_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.get_details(asset_id)

        """
        asset_id = _get_id_from_deprecated_uid(
            kwargs, asset_id, "asset", can_be_none=True
        )

        return self._get_asset_based_resource(
            asset_id,
            "data_asset",
            self._get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    async def aget_details(
        self,
        asset_id: str | None = None,
        get_all: bool | None = None,
        limit: int | None = None,
    ) -> dict:
        """Get data asset details asynchronously. If no asset_id is passed, details for all assets are returned.

        :param asset_id: unique ID of the asset
        :type asset_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = await client.data_assets.aget_details(asset_id)

        """
        return await self._aget_asset_based_resource(
            asset_id,
            "data_asset",
            self._get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    def create(
        self,
        name: str,
        file_path: str,
        duplicate_action: AssetDuplicateAction | None = None,
    ) -> dict[str, Any]:
        """Create a data asset and upload content to it.

        :param name: name to be given to the data asset
        :type name: str

        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :param duplicate_action: determines behaviour when asset with the same name already exists,
            if not specified, the value from catalogs/projects/spaces will be used
        :type duplicate_action: AssetDuplicateAction, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.create(name="sample_asset", file_path="/path/to/file")

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Assets._validate_type(name, "name", str, True)
        Assets._validate_type(file_path, "file_path", str, True)
        return self._create_asset(
            name,
            file_path,
            duplicate_action=(
                duplicate_action.value
                if isinstance(duplicate_action, Enum)
                else duplicate_action
            ),
        )

    async def acreate(
        self,
        name: str,
        file_path: str,
        duplicate_action: AssetDuplicateAction | None = None,
    ) -> dict[str, Any]:
        """Create a data asset and upload content to it asynchronously.

        :param name: name to be given to the data asset
        :type name: str

        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :param duplicate_action: determines behaviour when asset with the same name already exists,
            if not specified, the value from catalogs/projects/spaces will be used
        :type duplicate_action: AssetDuplicateAction, optional

        :return: metadata of the stored data asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = await client.data_assets.acreate(name="sample_asset", file_path="/path/to/file")

        """
        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Assets._validate_type(name, "name", str, True)
        Assets._validate_type(file_path, "file_path", str, True)
        return await self._acreate_asset(
            name,
            file_path,
            duplicate_action=(
                duplicate_action.value
                if isinstance(duplicate_action, Enum)
                else duplicate_action
            ),
        )

    def store(self, meta_props: dict) -> dict[str, Any]:
        """Create a data asset and upload content to it.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.data_assets.ConfigurationMetaNames.get()

        :type meta_props: dict

        **Example:**

        Example of data asset creation for files:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 'sample.csv'
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        Example of data asset creation using a connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '39eaa1ee-9aa4-4651-b8fe-95d3ddae',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1/sample.csv'
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        Example of data asset creation with a database sources type connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '23eaf1ee-96a4-4651-b8fe-95d3dadfe',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1'
            }
            asset_details = client.data_assets.store(meta_props=metadata)

        """
        name, file_path, description, connection_id, duplicate_action = (
            self._get_asset_details_from_metaprops(meta_props)
        )

        return self._create_asset(
            name,
            file_path,
            connection_id=connection_id,
            description=description,
            duplicate_action=(
                duplicate_action.value
                if isinstance(duplicate_action, Enum)
                else duplicate_action
            ),
        )

    async def astore(self, meta_props: dict) -> dict[str, Any]:
        """Create a data asset and upload content to it asynchronously.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.data_assets.ConfigurationMetaNames.get()

        :type meta_props: dict

        **Example:**

        Example of data asset creation for files:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 'sample.csv'
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        Example of data asset creation using a connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '39eaa1ee-9aa4-4651-b8fe-95d3ddae',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1/sample.csv'
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        Example of data asset creation with a database sources type connection:

        .. code-block:: python

            metadata = {
                client.data_assets.ConfigurationMetaNames.NAME: 'my data assets',
                client.data_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.data_assets.ConfigurationMetaNames.CONNECTION_ID: '23eaf1ee-96a4-4651-b8fe-95d3dadfe',
                client.data_assets.ConfigurationMetaNames.DATA_CONTENT_NAME: 't1'
            }
            asset_details = await client.data_assets.astore(meta_props=metadata)

        """
        name, file_path, description, connection_id, duplicate_action = (
            self._get_asset_details_from_metaprops(meta_props)
        )

        return await self._acreate_asset(
            name,
            file_path,
            connection_id=connection_id,
            description=description,
            duplicate_action=(
                duplicate_action.value
                if isinstance(duplicate_action, Enum)
                else duplicate_action
            ),
        )

    def _get_asset_details_from_metaprops(
        self, meta_props: dict
    ) -> tuple[str, str, str, str | None, str | None]:
        """Retrieves properties from meta_props for store and astore method returning them as a tuple in following order:
        name, file_path, description, connection_id, duplicate_action
        """
        Assets._validate_type(meta_props, "meta_props", dict, True)

        name = meta_props[self.ConfigurationMetaNames.NAME]
        file_path = meta_props[self.ConfigurationMetaNames.DATA_CONTENT_NAME]
        description = ""

        connection_id = meta_props.get(self.ConfigurationMetaNames.CONNECTION_ID)

        if not connection_id and not os.path.isfile(file_path):
            no_connection_id_specified_warning = (
                f"No connection_id specified and file: {file_path} does not exist."
            )
            warn(no_connection_id_specified_warning)

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            description = meta_props[self.ConfigurationMetaNames.DESCRIPTION]

        duplicate_action = meta_props.pop(
            self.ConfigurationMetaNames.DUPLICATE_ACTION, None
        )
        return name, file_path, description, connection_id, duplicate_action

    def _create_asset(
        self,
        name: str,
        file_path: str,
        connection_id: str | None = None,
        description: str | None = None,
        duplicate_action: str | None = None,
    ) -> dict:
        ##Step1: Create a data asset
        desc = description
        if desc is None:
            desc = ""
        try:
            import mimetypes
        except Exception as e:
            raise WMLClientError(
                Messages.get_message(message_id="module_mimetypes_not_found"), e
            )

        mime_type = mimetypes.MimeTypes().guess_type(file_path)[0]
        if mime_type is None:
            mime_type = "application/octet-stream"

        asset_meta: dict[str, Any] = {
            "metadata": {
                "name": name,
                "description": desc,
                "asset_type": "data_asset",
                "origin_country": "us",
                "asset_category": "USER",
            },
            "entity": {"data_asset": {"mime_type": mime_type}},
        }
        if connection_id is not None:
            asset_meta["metadata"]["tags"] = ["connected-data"]

        params = self._client._params()

        if duplicate_action:
            params["duplicate_action"] = duplicate_action

        # Step1  : Create an asset
        print(Messages.get_message(message_id="creating_data_asset"))

        creation_response = requests.post(
            self._client._href_definitions.get_data_assets_href(),
            headers=self._client._get_headers(),
            params=params,
            json=asset_meta,
        )

        asset_details = self._handle_response(
            201, "creating new asset", creation_response
        )
        # Step2: Create attachment

        asset_id = asset_details["metadata"]["asset_id"]
        attachment_name = file_path.split("/")[-1]
        attachment_meta: dict[str, Any] = {
            "asset_type": "data_asset",
            "name": attachment_name,
            "mime": mime_type,
        }
        if connection_id is not None:
            attachment_meta.update(
                {
                    "connection_id": connection_id,
                    "connection_path": file_path,
                    "is_remote": True,
                }
            )

        attachment_response = requests.post(
            self._client._href_definitions.get_attachments_href(asset_id),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        try:
            attachment_details = self._handle_response(
                201, "creating new attachment", attachment_response
            )
            if connection_id is None:
                attachment_id = attachment_details["attachment_id"]
                attachment_url = attachment_details["url1"]
                # Step3: Put content to attachment
                try:
                    with open(file_path, "rb") as _file:
                        if not self._client.ICP_PLATFORM_SPACES:
                            put_response = requests.put(attachment_url, data=_file)
                        else:
                            put_response = requests.put(
                                self._credentials.url + attachment_url,
                                files={"file": (name, _file, "file")},
                            )
                except Exception as e:
                    deletion_response = requests.delete(
                        self._client._href_definitions.get_data_asset_href(asset_id),
                        params=self._client._params(),
                        headers=self._client._get_headers(),
                    )
                    print(deletion_response.status_code)
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="failed_while_creating_a_data_asset"
                        ),
                        e,
                    )

                if put_response.status_code in {200, 201}:
                    # Step4: Complete attachment

                    complete_response = requests.post(
                        self._client._href_definitions.get_attachment_complete_href(
                            asset_id, attachment_id
                        ),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                    )

                    if complete_response.status_code == 200:
                        print(Messages.get_message(message_id="success"))
                        return self._get_required_element_from_response(asset_details)
                    else:
                        try:
                            self.delete(asset_id)
                        except Exception:
                            pass

                        if os.path.getsize(file_path) == 0:
                            raise WMLClientError(
                                Messages.get_message(
                                    message_id="cannot_create_empty_asset"
                                )
                            )

                        raise WMLClientError(
                            Messages.get_message(
                                message_id="failed_while_creating_a_data_asset"
                            )
                        )
                else:
                    try:
                        self.delete(asset_id)
                    except Exception:
                        pass
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="failed_while_creating_a_data_asset"
                        )
                    )
            else:
                print(Messages.get_message(message_id="success"))
                return self._get_required_element_from_response(asset_details)
        except ApiRequestFailure:
            try:
                self.delete(asset_id)
            except Exception:
                pass
            raise WMLClientError(
                Messages.get_message(message_id="failed_while_creating_a_data_asset")
            )

    async def _acreate_asset(
        self,
        name: str,
        file_path: str,
        connection_id: str | None = None,
        description: str | None = None,
        duplicate_action: str | None = None,
    ) -> dict:
        ##Step1: Create a data asset
        desc = description
        if desc is None:
            desc = ""
        try:
            import mimetypes
        except Exception as e:
            raise WMLClientError(
                Messages.get_message(message_id="module_mimetypes_not_found"), e
            )

        mime_type = mimetypes.MimeTypes().guess_type(file_path)[0]
        if mime_type is None:
            mime_type = "application/octet-stream"

        asset_meta: dict[str, Any] = {
            "metadata": {
                "name": name,
                "description": desc,
                "asset_type": "data_asset",
                "origin_country": "us",
                "asset_category": "USER",
            },
            "entity": {"data_asset": {"mime_type": mime_type}},
        }
        if connection_id is not None:
            asset_meta["metadata"]["tags"] = ["connected-data"]

        params = self._client._params()

        if duplicate_action:
            params["duplicate_action"] = duplicate_action

        # Step1  : Create an asset
        print(Messages.get_message(message_id="creating_data_asset"))

        creation_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_data_assets_href(),
            headers=await self._client._aget_headers(),
            params=params,
            json=asset_meta,
        )

        asset_details = self._handle_response(
            201, "creating new asset", creation_response
        )
        # Step2: Create attachment
        asset_id = asset_details["metadata"]["asset_id"]
        attachment_name = file_path.split("/")[-1]
        attachment_meta: dict[str, Any] = {
            "asset_type": "data_asset",
            "name": attachment_name,
            "mime": mime_type,
        }
        if connection_id is not None:
            attachment_meta.update(
                {
                    "connection_id": connection_id,
                    "connection_path": file_path,
                    "is_remote": True,
                }
            )

        attachment_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_attachments_href(asset_id),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=attachment_meta,
        )

        try:
            attachment_details = self._handle_response(
                201, "creating new attachment", attachment_response
            )
            if connection_id is None:
                attachment_id = attachment_details["attachment_id"]
                attachment_url = attachment_details["url1"]
                # Step3: Put content to attachment
                try:
                    if not self._client.ICP_PLATFORM_SPACES:
                        put_response = await self._client.async_httpx_client.put(
                            attachment_url, content=AsyncFileReader(file_path)
                        )
                    else:
                        with open(file_path, "rb") as _file:
                            put_response = await self._client.async_httpx_client.put(
                                self._credentials.url + attachment_url,
                                files={"file": (name, _file, "file")},
                            )
                except Exception as e:
                    deletion_response = await self._client.async_httpx_client.delete(
                        self._client._href_definitions.get_data_asset_href(asset_id),
                        params=self._client._params(),
                        headers=await self._client._aget_headers(),
                    )
                    print(deletion_response.status_code)
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="failed_while_creating_a_data_asset"
                        ),
                        e,
                    )

                if put_response.status_code in {200, 201}:
                    # Step4: Complete attachment

                    complete_response = await self._client.async_httpx_client.post(
                        self._client._href_definitions.get_attachment_complete_href(
                            asset_id, attachment_id
                        ),
                        headers=await self._client._aget_headers(),
                        params=self._client._params(),
                    )

                    if complete_response.status_code == 200:
                        print(Messages.get_message(message_id="success"))
                        return self._get_required_element_from_response(asset_details)
                    else:
                        try:
                            await self.adelete(asset_id)
                        except Exception:
                            pass

                        if os.path.getsize(file_path) == 0:
                            raise WMLClientError(
                                Messages.get_message(
                                    message_id="cannot_create_empty_asset"
                                )
                            )

                        raise WMLClientError(
                            Messages.get_message(
                                message_id="failed_while_creating_a_data_asset"
                            )
                        )
                else:
                    try:
                        await self.adelete(asset_id)
                    except Exception:
                        pass
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="failed_while_creating_a_data_asset"
                        )
                    )
            else:
                print(Messages.get_message(message_id="success"))
                return self._get_required_element_from_response(asset_details)
        except ApiRequestFailure:
            try:
                await self.adelete(asset_id)
            except Exception:
                pass
            raise WMLClientError(
                Messages.get_message(message_id="failed_while_creating_a_data_asset")
            )

    def list(self, limit: int | None = None) -> DataFrame:
        """Lists stored data assets in a table format.

        :param limit: limit number for fetched records
        :type limit: int, optional

        :rtype: DataFrame
        :return: listed elements

        **Example:**

        .. code-block:: python

            client.data_assets.list()

        """
        return self._list_asset_based_resource(
            url=self._client._href_definitions.get_search_data_asset_href(),
            column_names=["NAME", "ASSET_TYPE", "SIZE", "ASSET_ID"],
            limit=limit,
        )

    def download(
        self, asset_id: str | None = None, filename: str | None = None, **kwargs: Any
    ) -> str:  # asset_id is optional for backward compatibility,
        # filename should be not optional, however, as asset_id is, filename also must be
        """Download and store the content of a data asset.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :return: normalized path to the downloaded asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            client.data_assets.download(asset_id,"sample_asset.csv")

        """
        asset_id = _get_id_from_deprecated_uid(kwargs, asset_id, "asset")
        if filename is None:
            raise TypeError("Missing required positional argument 'filename'")

        content = self.get_content(asset_id)
        try:
            with open(filename, "wb") as f:
                f.write(content)
            print(
                Messages.get_message(
                    filename, message_id="successfully_saved_data_asset_content_to_file"
                )
            )
            return os.path.abspath(filename)
        except IOError as e:
            raise WMLClientError(
                Messages.get_message(
                    filename, message_id="saving_data_asset_to_local_file_failed"
                ),
                e,
            )

    async def adownload(self, asset_id: str, filename: str) -> str:
        """Download and store the content of a data asset asynchronously.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :return: normalized path to the downloaded asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            await client.data_assets.adownload(asset_id,"sample_asset.csv")

        """
        content = await self.aget_content(asset_id)
        try:
            with open(filename, "wb") as f:
                f.write(content)
            print(
                Messages.get_message(
                    filename, message_id="successfully_saved_data_asset_content_to_file"
                )
            )
            return os.path.abspath(filename)
        except IOError as e:
            raise WMLClientError(
                Messages.get_message(
                    filename, message_id="saving_data_asset_to_local_file_failed"
                ),
                e,
            )

    def get_content(
        self, asset_id: str | None = None, **kwargs: Any
    ) -> bytes:  # asset_id is optional for backward compatibility
        """Download the content of a data asset.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :return: the asset content
        :rtype: bytes

        **Example:**

        .. code-block:: python

            content = client.data_assets.get_content(asset_id).decode('ascii')

        """
        asset_id = _get_id_from_deprecated_uid(kwargs, asset_id, "asset")

        Assets._validate_type(asset_id, "asset_id", str, True)

        asset_response = requests.get(
            self._client._href_definitions.get_data_asset_href(asset_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        asset_details = self._handle_response(200, "get assets", asset_response)

        attachment_id = asset_details["attachments"][0]["id"]
        response = requests.get(
            self._client._href_definitions.get_attachment_href(asset_id, attachment_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if response.status_code != 200:
            raise WMLClientError(
                Messages.get_message(message_id="failure_during_downloading_data_asset")
            )

        if (
            "connection_id" in asset_details["attachments"][0]
            and asset_details["attachments"][0]["connection_id"] is not None
        ):
            conn_details = self._client.connections.get_details(
                asset_details["attachments"][0]["connection_id"]
            )
            attachment_data_source_type = conn_details["entity"].get("datasource_type")
            cos_conn_data_source_id = (
                self._client.connections.get_datasource_type_id_by_name(
                    "cloudobjectstorage"
                )
            )
            if attachment_data_source_type == cos_conn_data_source_id:
                attachment_signed_url = response.json()["url"]
                att_response = requests.get(attachment_signed_url)
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="download_api_not_supported_for_this_connection_type"
                    )
                )
        else:
            attachment_signed_url = response.json()["url"]
            if self._client.CLOUD_PLATFORM_SPACES:
                att_response = requests.get(attachment_signed_url)
            else:
                att_response = requests.get(
                    self._credentials.url + attachment_signed_url
                )

        if att_response.status_code != 200:
            raise ApiRequestFailure(
                Messages.get_message(
                    message_id="failure_during_downloading_data_asset"
                ),
                att_response,
            )

        return att_response.content

    async def aget_content(self, asset_id: str) -> bytes:
        """Download the content of a data asset asynchronously.

        :param asset_id: unique ID of the data asset to be downloaded
        :type asset_id: str

        :return: the asset content
        :rtype: bytes

        **Example:**

        .. code-block:: python

            content = (await client.data_assets.aget_content(asset_id)).decode('ascii')

        """

        Assets._validate_type(asset_id, "asset_id", str, True)

        asset_response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_data_asset_href(asset_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )
        asset_details = self._handle_response(200, "get assets", asset_response)

        attachment_id = asset_details["attachments"][0]["id"]
        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_attachment_href(asset_id, attachment_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        if response.status_code != 200:
            raise WMLClientError(
                Messages.get_message(message_id="failure_during_downloading_data_asset")
            )

        if (
            "connection_id" in asset_details["attachments"][0]
            and asset_details["attachments"][0]["connection_id"] is not None
        ):
            conn_details = await self._client.connections.aget_details(
                asset_details["attachments"][0]["connection_id"]
            )
            attachment_data_source_type = conn_details["entity"].get("datasource_type")
            cos_conn_data_source_id = await (
                self._client.connections.aget_datasource_type_id_by_name(
                    "cloudobjectstorage"
                )
            )
            if attachment_data_source_type == cos_conn_data_source_id:
                attachment_signed_url = response.json()["url"]
                att_response = await self._client.async_httpx_client.get(
                    attachment_signed_url
                )
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="download_api_not_supported_for_this_connection_type"
                    )
                )
        else:
            attachment_signed_url = response.json()["url"]
            if self._client.CLOUD_PLATFORM_SPACES:
                att_response = await self._client.async_httpx_client.get(
                    attachment_signed_url
                )
            else:
                att_response = await self._client.async_httpx_client.get(
                    self._credentials.url + attachment_signed_url
                )

        if att_response.status_code != 200:
            raise ApiRequestFailure(
                Messages.get_message(
                    message_id="failure_during_downloading_data_asset"
                ),
                att_response,
            )

        return att_response.content

    @staticmethod
    def get_id(asset_details: dict) -> str:
        """Get the unique ID of a stored data asset.

        :param asset_details: details of the stored data asset
        :type asset_details: dict

        :return: unique ID of the stored data asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.data_assets.get_id(asset_details)

        """
        Assets._validate_type(asset_details, "asset_details", object, True)
        Assets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "data_assets_details", ["metadata", "guid"]
        )

    @staticmethod
    def get_href(asset_details: dict) -> str:
        """Get the URL of a stored data asset.

        :param asset_details: details of the stored data asset
        :type asset_details: dict

        :return: href of the stored data asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_details = client.data_assets.get_details(asset_id)
            asset_href = client.data_assets.get_href(asset_details)

        """
        Assets._validate_type(asset_details, "asset_details", object, True)
        Assets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "asset_details", ["metadata", "href"]
        )

    def delete(
        self,
        asset_id: str | None = None,
        purge_on_delete: bool | None = None,
        **kwargs: Any,
    ) -> dict | str:  # asset_id is optional for backward compatibility
        """Soft delete the stored data asset. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.

        :param asset_id: unique ID of the data asset
        :type asset_id: str

        :param purge_on_delete: if set to True will purge the asset
        :type purge_on_delete: bool, optional

        :return: status ("SUCCESS" or "FAILED") or dictionary, if deleted asynchronously
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            client.data_assets.delete(asset_id)

        """
        return self._delete_asset_based_resource(
            asset_id,
            self._get_required_element_from_response,
            purge_on_delete,
            **kwargs,
        )

    async def adelete(
        self,
        asset_id: str,
        purge_on_delete: bool | None = None,
    ) -> dict | str:
        """Soft delete the stored data asset asynchronously. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.

        :param asset_id: unique ID of the data asset
        :type asset_id: str

        :param purge_on_delete: if set to True will purge the asset
        :type purge_on_delete: bool, optional

        :return: status ("SUCCESS" or "FAILED") or dictionary, if deleted asynchronously
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            await client.data_assets.adelete(asset_id)

        """
        return await self._adelete_asset_based_resource(
            asset_id,
            self._get_required_element_from_response,
            purge_on_delete,
        )

    def _get_required_element_from_response(self, response_data: dict) -> dict:
        WMLResource._validate_type(response_data, "data assets response", dict)

        import copy

        new_el = {"metadata": copy.copy(response_data["metadata"])}

        try:
            new_el["metadata"]["guid"] = response_data["metadata"]["asset_id"]
            new_el["metadata"]["href"] = response_data["href"]

            new_el["metadata"]["asset_type"] = response_data["metadata"]["asset_type"]
            new_el["metadata"]["created_at"] = response_data["metadata"]["created_at"]
            new_el["metadata"]["last_updated_at"] = response_data["metadata"][
                "usage"
            ].get("last_updated_at")

            if self._client.default_space_id is not None:
                new_el["metadata"]["space_id"] = response_data["metadata"]["space_id"]

            elif self._client.default_project_id is not None:
                new_el["metadata"]["project_id"] = response_data["metadata"][
                    "project_id"
                ]

            if "entity" in response_data:
                new_el["entity"] = response_data["entity"]

            if "attachments" in response_data and response_data["attachments"]:
                req_attachment_props = {"id", "name"}
                attachment_details = response_data["attachments"][0]

                new_el["metadata"].update(
                    {
                        f"attachment_{prop}": attachment_details[prop]
                        for prop in req_attachment_props
                        if prop in attachment_details
                    }
                )

            href_without_host = response_data["href"].split(".com")[-1]
            new_el["metadata"].update({"href": href_without_host})

            return new_el
        except Exception:
            raise WMLClientError(
                Messages.get_message(
                    response_data,
                    message_id="failed_to_read_response_from_down_stream_service",
                )
            )
