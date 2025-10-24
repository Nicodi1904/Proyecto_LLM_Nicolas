#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.metanames import RuntimeDefinitionsMetaNames
from ibm_watsonx_ai.wml_client_error import ResourceIdByNameNotFound
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai import APIClient


class RuntimeDefinitions(WMLResource):
    """Store and manage runtime definitions."""

    ConfigurationMetaNames = RuntimeDefinitionsMetaNames()
    """MetaNames for Runtime Definitions."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def store(self, meta_props: dict) -> dict:
        """Create a runtime definition.

        :param meta_props: metadata of the runtime definition configuration. To see available meta names, use:

            .. code-block:: python

                client.runtime_definitions.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the created runtime definition
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.runtime_definitions.ConfigurationMetaNames.NAME: "custom runtime definition",
                client.runtime_definitions.ConfigurationMetaNames.DESCRIPTION: "Custom runtime definition created with SDK",
             }

            client.runtime_definitions.store(meta_props)

        """
        self._client._check_if_either_is_set()

        RuntimeDefinitions._validate_type(meta_props, "meta_props", dict, True)
        runtime_definition_meta = (
            self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )
        )

        href = self._client._href_definitions.get_runtime_definitions_href()

        creation_response = requests.post(
            url=href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=runtime_definition_meta,
        )

        runtime_definition_details = self._handle_response(
            201, "creating runtime definition", creation_response
        )

        return runtime_definition_details

    def get_details(
        self, runtime_definition_id: str | None = None, include: str | None = None
    ) -> dict:
        """Get runtime definitions details.

        :param runtime_definition_id: unique ID of the runtime definitions
        :type runtime_definition_id: str

        :param include: query parameter to include additional fields, e.g., ``launch_configuration``.
            Requires cluster admin privileges.
        :type include: str, optional

        :return: metadata of the runtime definitions
        :rtype: dict

        **Example:**

        .. code-block:: python

            runtime_definition_details = client.runtime_definitions.get_details(runtime_definition_id)

        """
        self._client._check_if_either_is_set()

        RuntimeDefinitions._validate_type(
            runtime_definition_id, "runtime_definition_id", str, False
        )
        params = self._client._params(skip_space_project_chk=True)
        if include:
            params["include"] = include

        if runtime_definition_id is not None:
            url = self._client._href_definitions.get_runtime_definition_href(
                runtime_definition_id
            )
        else:
            url = self._client._href_definitions.get_runtime_definitions_href()

        response = requests.get(
            url=url,
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "get runtime definition details", response)

    def list(self, limit: int | None = None) -> DataFrame:
        """List runtime definitions in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed runtime definitions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.runtime_definitions.list()

        """

        runtime_definition_details = self.get_details()

        runtime_definition_values = [
            (
                m["entity"]["name"],
                m["metadata"]["guid"],
                m["entity"].get("description", ""),
            )
            for m in runtime_definition_details["resources"]
        ]
        table = self._list(
            runtime_definition_values, ["NAME", "ID", "DESCRIPTION"], limit
        )
        return table

    @staticmethod
    def get_id(runtime_definition_details: dict) -> str:
        """Get the ID of a runtime definition asset.

        :param runtime_definition_details: metadata of the runtime definition
        :type runtime_definition_details: dict

        :return: unique ID of the runtime definition
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.runtime_definitions.get_id(runtime_definition_details)

        """
        RuntimeDefinitions._validate_type(
            runtime_definition_details, "runtime_definition_details", object, True
        )

        return WMLResource._get_required_element_from_dict(
            runtime_definition_details,
            "runtime_definition_details",
            ["metadata", "guid"],
        )

    def get_id_by_name(self, runtime_definition_name: str) -> str:
        """Get the unique ID of a runtime definition for the given name.

        :param runtime_definition_name: name of the runtime definition
        :type runtime_definition_name: str

        :return: unique ID of the runtime definition
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.runtime_definitions.get_id_by_name(runtime_definition_name)

        """
        RuntimeDefinitions._validate_type(
            runtime_definition_name, "runtime_definition_name", str, True
        )

        runtime_definition_details = self.get_details()
        runtime_definition_id = [
            el["metadata"]["guid"]
            for el in runtime_definition_details["resources"]
            if el["entity"]["name"] == runtime_definition_name
        ]

        if runtime_definition_id:
            return runtime_definition_id[0]
        else:
            raise ResourceIdByNameNotFound(
                runtime_definition_name, "runtime definition"
            )

    def delete(self, runtime_definition_id: str) -> Literal["SUCCESS"]:
        """Delete a runtime definition.

        :param runtime_definition_id: unique ID of the runtime definition to be deleted
        :type runtime_definition_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.runtime_definitions.delete(runtime_definition_id)

        """
        self._client._check_if_either_is_set()

        RuntimeDefinitions._validate_type(
            runtime_definition_id, "runtime_definition_id", str, True
        )

        response = requests.delete(
            url=self._client._href_definitions.get_runtime_definition_href(
                runtime_definition_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete runtime definition", response)

    def update(self, runtime_definition_id: str, changes: dict) -> dict:
        """Updates existing runtime definition asset metadata.

        :param runtime_definition_id: ID of runtime definition to be updated
        :type runtime_definition_id: str

        :param changes: elements that will be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.runtime_definitions.ConfigurationMetaNames.NAME: "updated_runtime_definition"
            }

            runtime_definitions_details = client.runtime_definitions.update(
                runtime_definition_id,
                changes=metadata
            )

        """

        self._client._check_if_either_is_set()

        self._validate_type(runtime_definition_id, "runtime_definition_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self.get_details(
            runtime_definition_id, include="launch_configuration"
        )
        put_payload: dict = details["entity"]
        put_payload.update(changes)

        url = self._client._href_definitions.get_runtime_definition_href(
            runtime_definition_id
        )
        response = requests.put(
            url=url,
            json=put_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        return self._handle_response(200, "AI service patch", response)
