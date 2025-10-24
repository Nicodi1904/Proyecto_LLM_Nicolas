#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from ibm_watsonx_ai.foundation_models.extractions.text_extractions_v2_result_formats import (
    TextExtractionsV2ResultFormats,
)
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.utils.utils import get_from_json
from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    InvalidValue,
    UnexpectedType,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import pandas

    from ibm_watsonx_ai import APIClient, Credentials


class TextExtractionsV2(WMLResource):
    """Instantiate the Text Extraction service.

    :param credentials: credentials to the watsonx.ai instance
    :type credentials: Credentials, optional

    :param project_id: ID of the project, defaults to None
    :type project_id: str, optional

    :param space_id: ID of the space, defaults to None
    :type space_id: str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required, defaults to None
    :type api_client: APIClient, optional

    :raises InvalidMultipleArguments: raised when neither `api_client` nor `credentials` alongside `space_id` or `project_id` are provided
    :raises WMLClientError: raised if the CPD version is less than 5.1

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models.extractions import TextExtractionsV2

        extraction = TextExtractionsV2(
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        if credentials is not None:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials)
        elif api_client is not None:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id is not None:
            self._client.set.default_space(space_id)
        elif project_id is not None:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        if not self._client.CLOUD_PLATFORM_SPACES:
            cpd_version_error_message: str | None = None

            if self._client.CPD_version < 5.0:
                cpd_version_error_message = "Operation is unsupported for this release."
            elif self._client.CPD_version <= 5.1:
                cpd_version_error_message = (
                    f"For watsonx.ai software {self._client.CPD_version} release, please use "
                    "`ibm_watsonx_ai.foundation_models.extractions.TextExtractions` class."
                )

            if cpd_version_error_message:
                raise WMLClientError(cpd_version_error_message)

        super().__init__(__name__, self._client)

    def run_job(
        self,
        document_reference: DataConnection,
        results_reference: DataConnection,
        result_formats: (
            TextExtractionsV2ResultFormats
            | list[TextExtractionsV2ResultFormats]
            | list[str]
            | None
        ) = None,
        parameters: dict | None = None,
    ) -> dict:
        """Start a request to extract text and metadata from a document.

        :param document_reference: reference to the document in the bucket from which text will be extracted
        :type document_reference: DataConnection

        :param results_reference: reference to the location in the bucket where results will saved
        :type results_reference: DataConnection

        :param result_formats: result formats for the text extraction, can be passed as an enum or list, defaults to None
        :type result_formats: TextExtractionsV2ResultFormats | list[TextExtractionsV2ResultFormats] | list[str], optional

        :param parameters: the parameters for the text extraction, defaults to None
        :type parameters: dict | None, optional

        :return: raw response from the server with the text extraction job details
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.foundation_models.extractions import TextExtractionsV2ResultFormats
            from ibm_watsonx_ai.metanames import TextExtractionsV2ParametersMetaNames
            from ibm_watsonx_ai.helpers import DataConnection, S3Location

            document_reference = DataConnection(
                connection_asset_id="<connection_id>",
                location=S3Location(bucket="<bucket_name>", path="path/to/file"),
                )

            results_reference = DataConnection(
                connection_asset_id="<connection_id>",
                location=S3Location(bucket="<bucket_name>", path="path/to/directory/"),  # Path must end with /
                )

            response = extraction.run_job(
                document_reference=document_reference,
                results_reference=results_reference,
                parameters={
                    TextExtractionsV2ParametersMetaNames.MODE: "high_quality",
                    TextExtractionsV2ParametersMetaNames.OCR_MODE: "enabled",
                    TextExtractionsV2ParametersMetaNames.LANGUAGES: ["en", "fr"],
                    TextExtractionsV2ParametersMetaNames.AUTO_ROTATION_CORRECTION: True,
                    TextExtractionsV2ParametersMetaNames.CREATE_EMBEDDED_IMAGES: "enabled_placeholder",
                    TextExtractionsV2ParametersMetaNames.OUTPUT_DPI: 72,
                    TextExtractionsV2ParametersMetaNames.KVP_MODE: "invoice",
                    },
                result_formats=[
                    TextExtractionsV2ResultFormats.PLAIN_TEXT,
                    TextExtractionsV2ResultFormats.MARKDOWN,
                    TextExtractionsV2ResultFormats.ASSEMBLY_JSON,
                    ]
                )

        """

        if not isinstance(document_reference, DataConnection):
            raise UnexpectedType(
                el_name="document_reference",
                expected_type=DataConnection,
                actual_type=type(document_reference),
            )
        if not isinstance(results_reference, DataConnection):
            raise UnexpectedType(
                el_name="results_reference",
                expected_type=DataConnection,
                actual_type=type(results_reference),
            )

        if result_formats is None:
            result_formats = TextExtractionsV2ResultFormats.PLAIN_TEXT

        self._validate_type(parameters, "parameters", dict, False)

        payload = {
            "document_reference": document_reference.to_dict(),
            "results_reference": results_reference.to_dict(),
            "parameters": {
                "requested_outputs": (
                    [result_formats]
                    if isinstance(result_formats, TextExtractionsV2ResultFormats)
                    else result_formats
                ),
            },
        }

        if self._client.default_project_id is not None:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id is not None:
            payload["space_id"] = self._client.default_space_id

        if parameters is not None:
            payload["parameters"].update(parameters)

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_text_extractions_href(),
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(201, "run_job", response)

    def list_jobs(self, limit: int | None = None) -> pandas.DataFrame:
        """List text extraction jobs. If limit is None, all jobs will be listed.

        :param limit: limit number of fetched records, defaults to None
        :type limit: int | None, optional

        :return: text extraction jobs information as a pandas DataFrame
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            extraction.list_jobs()

        """
        import pandas

        columns = ["metadata.id", "metadata.created_at", "entity.results.status"]

        details = self.get_job_details(limit=limit)

        resources = details["resources"]
        data_normalize = pandas.json_normalize(resources)
        extraction_data = data_normalize.reindex(columns=columns)

        df_details = pandas.DataFrame(extraction_data, columns=columns)
        df_details.rename(
            columns={
                "metadata.id": "EXTRACTION_JOB_ID",
                "metadata.created_at": "CREATED",
                "entity.results.status": "STATUS",
            },
            inplace=True,
        )

        return df_details

    def get_job_details(
        self, extraction_job_id: str | None = None, limit: int | None = None
    ) -> dict:
        """Return text extraction job details. If `extraction_job_id` is None, return the details of all text extraction jobs.

        :param extraction_job_id: ID of the text extraction job, defaults to None
        :type extraction_job_id: str | None, optional

        :param limit: limit number of fetched records, defaults to None
        :type limit: int | None, optional

        :return: details of the text extraction job
        :rtype: dict

        **Example:**

        .. code-block:: python

            extraction.get_job_details(extraction_job_id="<extraction_job_id>")

        """
        self._validate_type(extraction_job_id, "extraction_job_id", str, False)

        if extraction_job_id is not None:
            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_text_extraction_href(
                    extraction_job_id
                ),
                params=self._client._params(skip_userfs=True),
                headers=self._client._get_headers(),
            )
        elif limit is None or 1 <= limit <= 200:
            params = self._client._params(skip_userfs=True)
            if limit is not None:
                params["limit"] = limit

            # TODO: pagination is not yet implemented
            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_text_extractions_href(),
                params=params,
                headers=self._client._get_headers(),
            )
        else:
            raise InvalidValue(
                value_name="limit",
                reason=f"The given value {limit} is not in between 1 and 200",
            )

        return self._handle_response(200, "get_job_details", response)

    def delete_job(self, extraction_job_id: str) -> Literal["SUCCESS"]:
        """Delete a text extraction job.

        :param extraction_job_id: ID of text extraction job
        :type extraction_job_id: str

        :return: "SUCCESS" if the deletion succeeds
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction.delete_job(extraction_job_id="<extraction_job_id>")

        """
        self._validate_type(extraction_job_id, "extraction_job_id", str, True)

        params = self._client._params(skip_userfs=True)
        params["hard_delete"] = True

        response = self._client.httpx_client.delete(
            url=self._client._href_definitions.get_text_extraction_href(
                extraction_job_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "delete_job", response)  # type: ignore[return-value]

    def cancel_job(self, extraction_job_id: str) -> Literal["SUCCESS"]:
        """Cancel a text extraction job.

        :param extraction_job_id: ID of text extraction job
        :type extraction_job_id: str

        :return: "SUCCESS" if the cancellation succeeds
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction.cancel_job(extraction_job_id="<extraction_job_id>")

        """
        self._validate_type(extraction_job_id, "extraction_job_id", str, True)

        response = self._client.httpx_client.delete(
            url=self._client._href_definitions.get_text_extraction_href(
                extraction_job_id
            ),
            params=self._client._params(skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "cancel_job", response)  # type: ignore[return-value]

    def get_results_reference(self, extraction_job_id: str) -> DataConnection:
        """Get a `DataConnection` instance that is a reference to the results stored on COS.

        :param extraction_job_id: ID of text extraction job
        :type extraction_job_id: str

        :return: location of the Data Connection to text extraction job results
        :rtype: DataConnection

        **Example:**

        .. code-block:: python

            results_reference = extraction.get_results_reference(extraction_job_id="<extraction_job_id>")

        """
        self._validate_type(extraction_job_id, "extraction_job_id", str, True)

        job_details = self.get_job_details(extraction_job_id)

        results_reference = get_from_json(job_details, ["entity", "results_reference"])
        data_conn = DataConnection._from_dict(results_reference)
        data_conn.set_client(self._client)
        return data_conn

    @classmethod
    def get_job_id(cls, extraction_details: dict) -> str:
        """Get the unique ID of a stored extraction request.

        :param extraction_details: metadata of the stored extraction
        :type extraction_details: dict

        :return: unique ID of the stored extraction request
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction_details = extraction.run_job(...)
            extraction_job_id = extraction.get_id(extraction_details)

        """
        cls._validate_type(extraction_details, "extraction_details", dict, True)

        return cast(
            str,
            cls._get_required_element_from_dict(
                extraction_details, "extraction_details", ["metadata", "id"]
            ),
        )
