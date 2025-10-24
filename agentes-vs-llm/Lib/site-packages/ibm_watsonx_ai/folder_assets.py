#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import FolderAssetsMetaNames
from ibm_watsonx_ai.utils import DATA_ASSETS_DETAILS_TYPE
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class FolderAssets(WMLResource):
    """Store and manage folder assets."""

    ConfigurationMetaNames = FolderAssetsMetaNames()
    """MetaNames for Folder Assets creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        folder_asset_id: str | None = None,
        get_all: bool | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get folder asset details. If no ``folder_asset_id`` is passed, details for all assets are returned.

        :param folder_asset_id: unique ID of the asset
        :type folder_asset_id: str

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :return: metadata of the stored folder asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            folder_asset_details = client.folder_assets.get_details(folder_asset_id)

        """
        folder_asset_id = _get_id_from_deprecated_uid(
            kwargs, folder_asset_id, "folder_asset", can_be_none=True
        )

        return self._get_asset_based_resource(
            folder_asset_id,
            "folder_asset",
            self._get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    def create(
        self,
        name: str,
        connection_path: str,
        connection_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a folder asset.

        :param name: name to be given to the folder asset
        :type name: str

        :param connection_path: path to the folder asset
        :type connection_path: str

        :param connection_id: ID of the connection where the folder asset is placed
        :type connection_id: str, optional

        :return: metadata of the stored folder asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            folder_asset_details = client.folder_assets.create(
                name="sample_folder_asset",
                connection_id="sample_connection_id",
                connection_path="/bucket1/folder1/folder1.1"
            )

        """
        FolderAssets._validate_type(name, "name", str, True)
        FolderAssets._validate_type(connection_path, "connection_path", str, True)
        FolderAssets._validate_type(connection_id, "connection_id", str, False)

        return self._create_asset(
            name=name,
            connection_path=connection_path,
            connection_id=connection_id,
        )

    def store(self, meta_props: dict) -> dict[str, Any]:
        """Create a folder asset.

        :param meta_props: metadata of the space configuration. To see available meta names, use:

            .. code-block:: python

                client.folder_assets.ConfigurationMetaNames.get()

        :type meta_props: dict

        **Example:**

        Example of creating a folder asset placed in a project/space container:

        .. code-block:: python

            metadata = {
                client.folder_assets.ConfigurationMetaNames.NAME: 'my folder asset',
                client.folder_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.folder_assets.ConfigurationMetaNames.CONNECTION_PATH: '/bucket1/folder1/folder1.1'
            }
            asset_details = client.folder_assets.store(meta_props=metadata)

        Example of creating a folder asset connected to a COS bucket folder:

        .. code-block:: python

            metadata = {
                client.folder_assets.ConfigurationMetaNames.NAME: 'my folder asset',
                client.folder_assets.ConfigurationMetaNames.DESCRIPTION: 'sample description',
                client.folder_assets.ConfigurationMetaNames.CONNECTION_ID: 'f1fea17c-a7e5-49e4-9f8e-23cef3e11ed5',
                client.folder_assets.ConfigurationMetaNames.CONNECTION_PATH: '/bucket1/folder1/folder1.1'
            }
            asset_details = client.folder_assets.store(meta_props=metadata)

        """
        self._client._check_if_either_is_set()

        FolderAssets._validate_type(meta_props, "meta_props", dict, True)
        FolderAssets._validate_meta_prop(meta_props, "name", str, True)
        FolderAssets._validate_meta_prop(meta_props, "connection_path", str, True)

        name = meta_props[self.ConfigurationMetaNames.NAME]
        connection_path = meta_props[self.ConfigurationMetaNames.CONNECTION_PATH]
        connection_id = meta_props.get(self.ConfigurationMetaNames.CONNECTION_ID)
        description = meta_props.get(self.ConfigurationMetaNames.DESCRIPTION, "")

        return self._create_asset(
            name=name,
            connection_path=connection_path,
            connection_id=connection_id,
            description=description,
        )

    def _create_asset(
        self,
        name: str,
        connection_path: str,
        connection_id: str | None = None,
        description: str | None = None,
    ) -> dict:
        # Step 1: Process payload
        desc = description or ""

        asset_meta: dict[str, Any] = {
            "metadata": {
                "name": name,
                "description": desc,
                "asset_type": "folder_asset",
                "origin_country": "us",
                "asset_category": "USER",
            },
            "entity": {"folder_asset": {"connection_path": connection_path}},
        }
        if connection_id is not None:
            asset_meta["entity"]["folder_asset"].update(
                {"connection_id": connection_id}
            )

        params = self._client._params()

        # Step 2: Create a folder asset
        print(Messages.get_message(message_id="creating_folder_asset"))

        creation_response = requests.post(
            self._client._href_definitions.get_folder_assets_href(),
            headers=self._client._get_headers(),
            params=params,
            json=asset_meta,
        )

        asset_details = self._handle_response(
            201, "creating new folder asset", creation_response
        )

        if creation_response.status_code == 201:
            print(Messages.get_message(message_id="success"))
            return self._get_required_element_from_response(asset_details)
        else:
            raise WMLClientError(
                Messages.get_message(message_id="failed_while_creating_a_folder_asset")
            )

    def list(self, limit: int | None = None) -> DataFrame:
        """Lists stored folder assets in a table format.

        :param limit: limit number for fetched records
        :type limit: int, optional

        :rtype: DataFrame
        :return: listed elements

        **Example:**

        .. code-block:: python

            client.folder_assets.list()

        """
        return self._list_asset_based_resource(
            url=self._client._href_definitions.get_search_folder_asset_href(),
            column_names=["NAME", "ASSET_TYPE", "ASSET_ID"],
            limit=limit,
        )

    @staticmethod
    def get_id(asset_details: dict) -> str:
        """Get the unique ID of a stored folder asset.

        :param asset_details: details of the stored folder asset
        :type asset_details: dict

        :return: unique ID of the stored folder asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.folder_assets.get_id(asset_details)

        """
        FolderAssets._validate_type(asset_details, "asset_details", object, True)
        FolderAssets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "folder_assets_details", ["metadata", "guid"]
        )

    @staticmethod
    def get_href(asset_details: dict) -> str:
        """Get the URL of a stored folder asset.

        :param asset_details: details of the stored folder asset
        :type asset_details: dict

        :return: href of the stored folder asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_details = client.folder_assets.get_details(asset_id)
            asset_href = client.folder_assets.get_href(asset_details)

        """
        FolderAssets._validate_type(asset_details, "asset_details", object, True)
        FolderAssets._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "asset_details", ["metadata", "href"]
        )

    def delete(
        self,
        asset_id: str,
        purge_on_delete: bool | None = None,
        **kwargs: Any,
    ) -> dict | str:
        """Soft delete the stored folder asset. The asset will be moved to trashed assets
        and will not be visible in asset list. To permanently delete assets set `purge_on_delete` parameter to True.

        :param asset_id: unique ID of the folder asset
        :type asset_id: str

        :param purge_on_delete: if set to True will purge the asset
        :type purge_on_delete: bool, optional

        :return: status ("SUCCESS" or "FAILED") or dictionary, if deleted asynchronously
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            client.folder_assets.delete(asset_id)

        """
        return self._delete_asset_based_resource(
            asset_id,
            self._get_required_element_from_response,
            purge_on_delete,
            **kwargs,
        )

    def _get_required_element_from_response(self, response_data: dict) -> dict:
        WMLResource._validate_type(response_data, "folder assets response", dict)

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
                new_el["metadata"].update(
                    {"attachment_id": response_data["attachments"][0]["id"]}
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
