#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any, Literal

import pandas

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource


class VectorIndexes(WMLResource):
    """Initiate the Vector Indexes class.

    :param api_client: instance of APIClient with default project_id set
    :type api_client: APIClient

    **Example:**

        .. code-block:: python

            vector_indexes = VectorIndexes(api_client=api_client)

    """

    def __init__(self, api_client: APIClient) -> None:
        if api_client.default_project_id is None:
            raise WMLClientError(
                error_msg=(
                    "Vector Index Assets are available only in project scope. "
                    "Please set project_id: `api_client.set.default_project('<project_id>')`"
                )
            )

        WMLResource.__init__(self, __name__, api_client)

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.1:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

    def create(
        self,
        name: str,
        description: str | None = None,
        store: dict | None = None,
        settings: dict | None = None,
        tags: list[str] | None = None,
        sample_questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Creates a new Vector Index Asset.

        :param name: name for vector index asset
        :type name: str

        :param description: optional description for the vector index asset, defaults to None
        :type description: str | None

        :param store: store parameters, defaults to None
        :type store: dict | None, optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict | None, optional

        :param tags: tags attached to the asset, defaults to None
        :type tags: list[str] | None, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str] | None, optional

        :param data_assets: IDs of the associated data assets used in the vector index, defaults to None
        :type data_assets: list[str] | None, optional

        :param build: the associated build to process the data for external vector stores, defaults to None
        :type build: dict | None, optional

        :param status: the status of the vector index, defaults to None
        :type status: str | None, optional

        :return: metadata of the created Vector Index Asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            params = dict(
                name="test_sdk_vector_index",
                description="Description",
                settings={
                    "embedding_model_id": "<model_id>",
                    "top_k": 1,
                    "schema_fields": {"text": "text"},
                },
                store={
                    "type": "watsonx.data",
                    "connection_id": "<connection_id>",
                    "index": "<index_name>",
                    "new_index": False,
                    "database": "default",
                },
                tags=["test_tag"],
                sample_questions=["Sample question"],
                status="ready",
            )

            vector_index_details = vector_indexes.create(**params)

        """

        WMLResource._validate_type(name, "name", str)
        WMLResource._validate_type(description, "description", str, False)
        WMLResource._validate_type(store, "store", dict, False)
        WMLResource._validate_type(settings, "settings", dict, False)

        data_assets = kwargs.pop("data_assets", None)
        WMLResource._validate_type(data_assets, "data_assets", list, False)

        build = kwargs.pop("build", None)
        WMLResource._validate_type(build, "build", dict, False)

        status = kwargs.pop("status", None)
        WMLResource._validate_type(status, "status", str, False)

        payload = self._prepare_payload_create(
            name=name,
            description=description,
            data_assets=data_assets,
            store=store,
            settings=settings,
            build=build,
            tags=tags,
            sample_questions=sample_questions,
            status=status,
        )

        response = self._client.httpx_client.post(
            url=self._client.service_instance._href_definitions.get_vector_indexes_href(),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(201, "create asset", response)

    @staticmethod
    def _prepare_payload_create(
        name: str | None = None,
        description: str | None = None,
        data_assets: list[str] | None = None,
        store: dict | None = None,
        settings: dict | None = None,
        build: dict | None = None,
        tags: list[str] | None = None,
        sample_questions: list[str] | None = None,
        status: str | None = None,
    ) -> dict:
        """Prepare payload based on the provided data."""
        payload: dict[str, Any] = {}

        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if data_assets is not None:
            payload["data_assets"] = data_assets
        if store is not None:
            payload["store"] = store
        if settings is not None:
            payload["settings"] = settings
        if build is not None:
            payload["build"] = build
        if tags is not None:
            payload["tags"] = tags
        if sample_questions is not None:
            payload["sample_questions"] = sample_questions
        if status is not None:
            payload["status"] = status

        return payload

    def update(
        self,
        index_id: str,
        name: str | None = None,
        description: str | None = None,
        store: dict | None = None,
        settings: dict | None = None,
        tags: list[str] | None = None,
        sample_questions: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update a Vector Index Asset with the given id.

        :param index_id: Vector Index ids
        :type index_id: str

        :param name: name for vector index asset, defaults to None
        :type name: str | None

        :param description: optional description for the vector index asset, defaults to None
        :type description: str | None

        :param store: store parameters, defaults to None
        :type store: dict | None, optional

        :param settings: settings of vector index, defaults to None
        :type settings: dict | None, optional

        :param tags: tags attached to the asset, defaults to None
        :type tags: list[str] | None, optional

        :param sample_questions: sample asked questions, defaults to None
        :type sample_questions: list[str] | None, optional

        :param data_assets: IDs of the associated data assets used in the vector index, defaults to None
        :type data_assets: list[str] | None, optional

        :param build: the associated build to process the data for external vector stores, defaults to None
        :type build: dict | None, optional

        :param status: the status of the vector index, defaults to None
        :type status: str | None, optional

        :return: metadata of the created Vector Index Asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_indexes.update(index_id, name="new_name", description="new_description")

        """

        WMLResource._validate_type(name, "name", str, False)
        WMLResource._validate_type(description, "description", str, False)
        WMLResource._validate_type(store, "store", dict, False)
        WMLResource._validate_type(settings, "settings", dict, False)

        data_assets = kwargs.pop("data_assets", None)
        WMLResource._validate_type(data_assets, "data_assets", list, False)

        build = kwargs.pop("build", None)
        WMLResource._validate_type(build, "build", dict, False)

        status = kwargs.pop("status", None)
        WMLResource._validate_type(status, "status", str, False)

        payload = self._prepare_payload_create(
            name=name,
            description=description,
            data_assets=data_assets,
            store=store,
            settings=settings,
            build=build,
            tags=tags,
            sample_questions=sample_questions,
            status=status,
        )

        response = self._client.httpx_client.patch(
            url=self._client.service_instance._href_definitions.get_vector_index_href(
                index_id
            ),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "update asset", response)

    def get_details(self, index_id: str) -> dict:
        """Get details of Vector Index Asset with given `index_id`.

        :param index_id: Vector Index id
        :type index_id: str

        :return: details of Vector Index Asset with given index_id
        :rtype: dict

        **Example:**

        .. code-block:: python

            vector_indexes.get_details(index_id)

        """
        response = self._client.httpx_client.get(
            url=self._client.service_instance._href_definitions.get_vector_index_href(
                index_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "create asset", response)

    def delete(self, index_id: str) -> Literal["SUCCESS"]:
        """Delete a vector index with the given id.

        :param index_id: Vector Index id
        :type index_id: str

        :return: "SUCCESS" if delete successfully
        :rtype: str

        **Example:**

        .. code-block:: python

            vector_indexes.delete(index_id)

        """

        response = self._client.httpx_client.delete(
            url=self._client.service_instance._href_definitions.get_vector_index_href(
                index_id
            ),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete", response)  # type: ignore[return-value]

    def _get_details(self, limit: int | None = None) -> list:
        """Get details of all vector indexes. If limit is set to None,
        then all vector indexes are fetched.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: int | None

        :return: List of vector indexes metadata
        :rtype: list
        """
        headers = self._client._get_headers()
        url = self._client.service_instance._href_definitions.get_vector_indexes_all_href()
        json_data: dict[str, int | str] = {
            "query": "asset.asset_type:vector_index",
            "sort": "-asset.created_at<string>",
        }
        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            json_data.update({"limit": limit})
        else:
            json_data.update({"limit": 200})
        vector_indexes_list = []
        bookmark = True
        while bookmark is not None:
            response = self._client.httpx_client.post(
                url=url, json=json_data, headers=headers, params=self._client._params()
            )

            details_json = self._handle_response(200, "Get next details", response)
            bookmark = details_json.get("next", {"href": None}).get("bookmark", None)
            vector_indexes_list.extend(details_json.get("results", []))
            if limit is not None:
                break
            json_data.update({"bookmark": bookmark})
        return vector_indexes_list

    def list(self, *, limit: int | None = None) -> pandas.DataFrame:
        """List all available Vector Index Assets in the DataFrame format.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: int, optional

        :return: DataFrame of fundamental properties of available Vector Index Assets.
        :rtype: pandas.core.frame.DataFrame

        **Example:**

        .. code-block:: python

            vector_indexes.list(limit=5)    # list of 5 recently created vector index assets

        """

        details = [
            "metadata.asset_id",
            "metadata.name",
            "metadata.created_at",
            "metadata.usage.last_updated_at",
        ]
        prompts_details = self._get_details(limit=limit)

        data_normalize = pandas.json_normalize(prompts_details)
        prompts_data = data_normalize.reindex(columns=details)

        df_details = pandas.DataFrame(prompts_data, columns=details)

        df_details.rename(
            columns={
                "metadata.asset_id": "ID",
                "metadata.name": "NAME",
                "metadata.created_at": "CREATED",
                "metadata.usage.last_updated_at": "LAST MODIFIED",
            },
            inplace=True,
        )

        return df_details
