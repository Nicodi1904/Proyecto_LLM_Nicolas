#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

__all__ = ["BaseDataConnection"]

import io
import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Tuple, Union
from warnings import warn

import pandas as pd
from pandas import DataFrame, read_csv

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.data_loaders.datasets.experiment import (
    DEFAULT_SAMPLE_SIZE_LIMIT,
    DEFAULT_SAMPLING_TYPE,
)
from ibm_watsonx_ai.utils.autoai.enums import DataConnectionTypes
from ibm_watsonx_ai.utils.autoai.errors import (
    CannotReadSavedRemoteDataBeforeFit,
    ContainerTypeNotSupported,
    NotS3Connection,
)
from ibm_watsonx_ai.utils.autoai.utils import (
    _error_on_duplicate_columns_csv,
    try_load_dataset,
    try_load_tar_gz,
)
from ibm_watsonx_ai.utils.utils import get_from_json, is_lib_installed
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    DataStreamError,
)

if TYPE_CHECKING:
    from ibm_boto3 import resource

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.helpers.connections import DataConnection


class BaseDataConnection(ABC):
    """Base class for DataConnection."""

    def __init__(self):
        self.type = None
        self.connection = None
        self.location = None
        self.auto_pipeline_params = None
        self._api_client = None
        self._run_id = None
        self.id = None
        self._datasource_type = None

    @abstractmethod
    def _to_dict(self) -> dict:
        """Convert DataConnection object to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, _dict: dict) -> "BaseDataConnection":
        """Create a DataConnection object from dictionary."""
        pass

    @abstractmethod
    def read(
        self, with_holdout_split: bool = False
    ) -> Union["DataFrame", Tuple["DataFrame", "DataFrame"]]:
        """Download dataset stored in remote data storage."""
        pass

    @abstractmethod
    def write(self, data: Union[str, "DataFrame"], remote_name: str) -> None:
        """Upload file to a remote data storage."""
        pass

    def _fill_experiment_parameters(
        self,
        prediction_type: str,
        prediction_column: str,
        holdout_size: float,
        csv_separator: str = ",",
        excel_sheet: Union[str, int] = None,
        encoding: str = "utf-8",
    ) -> None:
        """To be able to recreate a holdout split, this method need to be called."""
        self.auto_pipeline_params = {
            "prediction_type": prediction_type,
            "prediction_column": prediction_column,
            "holdout_size": holdout_size,
            "csv_separator": csv_separator,
            "excel_sheet": excel_sheet,
            "encoding": encoding,
        }

    def _download_csv_file(self, path) -> dict:
        """Download csv file."""
        df = pd.DataFrame()

        if "//" in path:  # sometimes there is an additional slash, need to replace it
            path = path.replace("//", "/")

        if self.type == DataConnectionTypes.FS:
            csv_url = (
                self._api_client._href_definitions.get_wsd_model_attachment_href()
                + f"/auto_ml/{path.split('/auto_ml/')[-1]}"
            )

            with requests.get(
                csv_url,
                params=self._api_client._params(),
                headers=self._api_client._get_headers(),
                stream=True,
            ) as file_response:
                if file_response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading json"), file_response
                    )

                downloaded_asset = file_response.content
                df = pd.read_csv(io.BytesIO(downloaded_asset))
                # json_content = json.loads(buffer.getvalue().decode('utf-8'))

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if self._check_if_connection_asset_is_s3():
                from .connections import _AmazonS3Connection

                if isinstance(self.connection, _AmazonS3Connection):
                    raise NotImplementedError(
                        "The operation is not supported for AmazonS3 connection. Try with Flight Service enabled"
                    )
                cos_client = self._init_cos_client()

                try:
                    file = cos_client.Object(self.location.bucket, path).get()
                    content = file["Body"].read()
                    df = pd.read_csv(io.BytesIO(content))

                except Exception as cos_access_exception:
                    raise ConnectionError(
                        f"Unable to access data object in cloud object storage with credentials supplied. "
                        f"Error: {cos_access_exception}"
                    )

            else:
                raise NotImplementedError(
                    f"Unsupported connection type: {self.type}. "
                    f"Datasource type is not supported. "
                    f"Supported type is: bluemixcloudobjectstorage"
                )

        else:
            raise NotImplementedError(f"Unsupported connection type: {self.type}")

        return df

    def _download_json_file(
        self,
        path,
        tuning_type: (
            Literal["prompt_tuning", "fine_tuning", "ilab_tuning"] | None
        ) = None,
    ) -> dict:
        """Download json file."""
        json_content = {}

        if "//" in path:  # sometimes there is an additional slash, need to replace it
            path = path.replace("//", "/")

        # TODO: remove S3 implementation
        if self.type == DataConnectionTypes.S3:
            s3_dataconnection_deprecated_warning = (
                "S3 DataConnection is deprecated! Please use data_asset_id instead."
            )
            warn(s3_dataconnection_deprecated_warning, category=DeprecationWarning)

            cos_client = self._init_cos_client()

            try:
                file = cos_client.Object(self.location.bucket, path).get()
                content = file["Body"].read()
                json_content = json.loads(content.decode("utf-8"))
            except Exception as cos_access_exception:
                raise ConnectionError(
                    f"Unable to access data object in cloud object storage with credentials supplied. "
                    f"Error: {cos_access_exception}"
                )
        elif self.type == DataConnectionTypes.FS:
            if tuning_type == "prompt_tuning":
                json_url = (
                    self._api_client._href_definitions.get_wsd_model_attachment_href()
                    + f"wx_prompt_tune/{path.split('/wx_prompt_tune/')[-1]}"
                )
            elif tuning_type == "fine_tuning":
                json_url = (
                    self._api_client._href_definitions.get_wsd_model_attachment_href()
                    + f"wx_fine_tune/{path.split('/wx_fine_tune/')[-1]}"
                )
            else:
                json_url = (
                    self._api_client._href_definitions.get_wsd_model_attachment_href()
                    + f"/auto_ml/{path.split('/auto_ml/')[-1]}"
                )

            with requests.get(
                json_url,
                params=self._api_client._params(),
                headers=self._api_client._get_headers(),
                stream=True,
            ) as file_response:
                if file_response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading json"), file_response
                    )

                downloaded_asset = file_response.content
                buffer = io.BytesIO(downloaded_asset)
                try:
                    json_content = json.loads(buffer.getvalue().decode("utf-8"))
                except json.JSONDecodeError:
                    json_content = [
                        json.loads(jline)
                        for jline in buffer.getvalue().decode("utf-8").splitlines()
                    ]

        elif self.type == DataConnectionTypes.CA or self.type == DataConnectionTypes.CN:
            if self._check_if_connection_asset_is_s3():
                cos_client = self._init_cos_client()
                data_conn = self.to_dict()
                bucket = get_from_json(
                    data_conn, ["location", "bucket"]
                ) or get_from_json(
                    data_conn,
                    ["connection", "bucket"],  # AWS containers has bucket in connection
                )

                if bucket is None:
                    raise ValueError(
                        "Missing bucket in connection or location of the DataConnection object."
                    )

                try:
                    file = cos_client.Object(bucket, path).get()
                    content = file["Body"].read()
                    try:
                        json_content = json.loads(content.decode("utf-8"))
                    except json.JSONDecodeError:
                        json_content = [
                            json.loads(jline)
                            for jline in content.decode("utf-8").splitlines()
                        ]

                except Exception as cos_access_exception:
                    raise ConnectionError(
                        f"Unable to access data object in cloud object storage with credentials supplied. "
                        f"Error: {cos_access_exception}"
                    )

            else:
                raise NotImplementedError(
                    f"Unsupported connection type: {self.type}. "
                    f"Datasource type is not supported. "
                    f"Supported type is: bluemixcloudobjectstorage"
                )

        else:
            raise NotImplementedError(f"Unsupported connection type: {self.type}")

        return json_content

    @staticmethod
    def _get_attachment_details(data_asset_id: str, api_client: APIClient) -> dict:
        response = requests.get(
            api_client._href_definitions.get_data_asset_href(data_asset_id),
            params=api_client._params(),
            headers=api_client._get_headers(),
        )
        data_asset_details = api_client.data_assets._handle_response(
            200, "GET data asset details", response
        )

        attachments_data_asset_details = data_asset_details.get("attachments", [{}])[0]
        # note: Return attachment details from data asset details if it is not a connected data asset:
        if (
            attachments_data_asset_details
            and attachments_data_asset_details.get("connection_id") is None
        ):
            return attachments_data_asset_details
        else:
            attachment_id = attachments_data_asset_details.get("id")
            response = requests.get(
                api_client._href_definitions.get_attachment_href(
                    data_asset_id, attachment_id
                ),
                params=api_client._params(),
                headers=api_client._get_headers(),
                verify=False if api_client.ICP_PLATFORM_SPACES else None,
            )

            api_client.data_assets._handle_response(
                200, "GET attachment details", response
            )

            return response.json()

    def _prepare_connection_details(self) -> dict:
        connection_details = {}

        if self.type == DataConnectionTypes.CA:
            if self._api_client is not None:
                connection_details = self._api_client.connections.get_details(
                    self.connection.id
                )
            else:
                try:
                    from ibm_watson_studio_lib import access_project_or_space

                    wslib = access_project_or_space()
                    connection_details = wslib.by_id.get_connection(self.connection.id)

                except ModuleNotFoundError:
                    raise NotImplementedError(
                        "This functionality can be run only on Watson Studio."
                    )

        elif self.type == DataConnectionTypes.CN:
            connection_details = self._create_conn_details_for_container()

        else:
            raise NotS3Connection(_internal=True)

        return connection_details

    def _is_shared_bucket(self) -> bool:
        try:
            connection_details = self._prepare_connection_details()
        except NotS3Connection:
            return False

        return get_from_json(
            connection_details, ["entity", "properties", "shared"], False
        )

    def _check_if_connection_asset_is_s3(self) -> bool:
        try:
            # fast return true if the connection is an instance of S3Connection and non s3 type,
            # or its attribute "is_s3" takes True
            from .connections import S3Connection

            if (
                isinstance(self.connection, S3Connection)
                and self.type != DataConnectionTypes.S3
                and self.connection.to_dict()
            ) or (
                self.connection is not None and getattr(self.connection, "is_s3", False)
            ):
                return True

            if self.type in {
                DataConnectionTypes.S3,
                DataConnectionTypes.FS,
                DataConnectionTypes.DS,
            }:
                return False

            try:
                connection_details = self._prepare_connection_details()
            except NotS3Connection:
                return False

            # Note: Check with project libs if connection points to S3 (COS or AWS)
            if self._api_client is not None:
                datasource_type = connection_details["entity"]["datasource_type"]
                self._datasource_type = datasource_type
                datasource_type_id_ibm_cos = (
                    self._api_client.connections.get_datasource_type_id_by_name(
                        "bluemixcloudobjectstorage"
                    )
                )
                datasource_type_id_aws_cos = (
                    self._api_client.connections.get_datasource_type_id_by_name(
                        "cloudobjectstorage"
                    )
                )

                if self._datasource_type in {
                    datasource_type_id_ibm_cos,
                    datasource_type_id_aws_cos,
                    "bluemixcloudobjectstorage",
                    "cloudobjectstorage",
                    "amazons3",
                }:
                    is_s3 = True

                else:
                    is_s3 = False

            elif self.type == DataConnectionTypes.CN:
                is_s3 = True

            elif "url" in connection_details:
                is_s3 = True
                connection_details["entity"] = {"properties": connection_details}

            else:
                is_s3 = False
            # --- end note

            if is_s3:
                if self._datasource_type == "amazons3":
                    self._init_s3_aws_connection(connection_details)
                else:
                    self._init_s3_cos_connection(connection_details)

            return is_s3

        except Exception as e:
            if (
                os.environ.get("USER_ACCESS_TOKEN") is None
                and os.environ.get("RUNTIME_ENV_ACCESS_TOKEN_FILE") is None
            ):
                raise e

            else:
                return False  # if we are in WS, ignore this check even if there was some error

    def _init_s3_cos_connection(self, connection_details: dict) -> None:
        """
        Helper function that initializes internal `S3Connection` object based on connection_details retrieved
         from connection asset or container (IBM Cloud).
        """
        connection_props = connection_details["entity"]["properties"]

        connection_values = {
            "api_key": connection_props.get("api_key"),
            "auth_endpoint": connection_props.get("iam_url"),
            "endpoint_url": connection_props.get("url"),
            "resource_instance_id": connection_props.get("resource_instance_id"),
            "access_key_id": connection_props.get("access_key"),
            "secret_access_key": connection_props.get("secret_key"),
        }

        if "cos_hmac_keys" in str(connection_props):
            creds = json.loads(connection_props["credentials"])
            connection_values["access_key"] = creds["cos_hmac_keys"]["access_key_id"]
            connection_values["secret_key"] = creds["cos_hmac_keys"][
                "secret_access_key"
            ]

        if self.connection is None:
            from .connections import S3Connection

            self.connection = S3Connection(
                **connection_values,
                _internal_use=True,
            )

        else:
            for key, val in connection_values.items():
                if val is not None:
                    setattr(self.connection, key, val)

        if not ("api_key" in connection_props and "iam_url" in connection_props):
            [
                delattr(self.connection, attr)
                for attr in ["auth_endpoint", "resource_instance_id", "api_key"]
                if hasattr(self.connection, attr)
            ]

        if self.type == DataConnectionTypes.CN:
            self.location.bucket = connection_props["bucket_name"]

        self.connection.is_s3 = True

    def _init_s3_aws_connection(self, connection_details: dict) -> None:
        """
        Helper function that initializes internal `_AmazonS3Connection` object
        based on `connection_details` retrieved from container (IBM Cloud on AWS).
        """
        from .connections import _AmazonS3Connection

        connection_props = connection_details["entity"]["properties"]

        self.connection = _AmazonS3Connection(
            access_key=connection_props["credentials"]["access_key_id"],
            bucket=connection_props["bucket_name"],
            region=connection_props["bucket_region"],
            secret_key=connection_props["credentials"]["secret_access_key"],
            session_token=connection_props["credentials"]["session_token"],
            shared_credentials=connection_props.get("shared", True),
        )

        if (
            self.type == DataConnectionTypes.CN
            and self.connection._shared_credentials
            and self._api_client is not None
        ):
            container_path_prefix = (
                self._api_client.default_project_id or self._api_client.default_space_id
            )
            if container_path_prefix and not self.location.path.startswith(
                container_path_prefix
            ):
                self.location.path = f"{container_path_prefix}/{self.location.path}"

    def _is_data_asset_normal(self) -> bool:
        """Returns `True` if data asset is normal data asset - not connected data asset."""
        try:
            if self.type == "data_asset":
                items = (
                    self.connection.href.split("/")
                    if self.connection is not None and hasattr(self.connection, "href")
                    else self.location.href.split("/")
                )

                data_asset_id = items[-1].split("?")[0]

                if self._api_client is not None:
                    attachment_details = self._get_attachment_details(
                        data_asset_id, self._api_client
                    )
                    return bool("connection_id" not in attachment_details)
                else:
                    try:
                        from ibm_watson_studio_lib import access_project_or_space

                        wslib = access_project_or_space()

                        # note: Check if asset is located directly in the project files
                        #       If yes it is not a connected data.
                        #       [Prevents unnecessary logging].
                        return any(
                            file["asset_id"] == data_asset_id
                            for file in wslib.list_stored_data()
                        )
                        # --- end note

                    except ModuleNotFoundError:
                        raise NotImplementedError(
                            "This functionality can be run only on Watson Studio."
                        )

        except Exception as e:
            if (
                os.environ.get("USER_ACCESS_TOKEN") is None
                and os.environ.get("RUNTIME_ENV_ACCESS_TOKEN_FILE") is None
            ):
                raise e
            else:
                return False  # if we are in WS, ignore this check even if there was some error

    def _is_data_asset_nfs(self):
        if self.type == "data_asset":
            items = (
                self.connection.href.split("/")
                if self.connection is not None and hasattr(self.connection, "href")
                else self.location.href.split("/")
            )

            data_asset_id = items[-1].split("?")[0]

            if self._api_client is not None:
                attachment_details = self._get_attachment_details(
                    data_asset_id, self._api_client
                )

                return "connection_id" in attachment_details and attachment_details.get(
                    "datasource_type"
                ) == self._api_client.connections.get_datasource_type_id_by_name(
                    "volumes"
                )
        return False

    def _download_indices_from_cos(
        self, cos_client: "resource", location_path
    ) -> "DataFrame":
        """Download indices for this connection. COS version"""

        try:
            file = cos_client.Object(self.location.bucket, location_path).get()
        except Exception:
            file = list(
                cos_client.Bucket(self.location.bucket).objects.filter(
                    Prefix=location_path
                )
            )[0].get()

        buffer = io.BytesIO(file["Body"].read())

        if ".csv" in location_path:
            file_name = "indices.csv"
            with open(file_name, "wb") as out:
                out.write(buffer.read())

            data = read_csv(
                file_name,
                sep=self.auto_pipeline_params.get("csv_separator", ","),
                encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
            )

        else:
            data = try_load_tar_gz(
                buffer=buffer,
                separator=self.auto_pipeline_params.get("csv_separator", ","),
                encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
            )

        return data

    def _download_training_data_from_data_asset_storage(
        self,
        binary: bool = False,
        is_flight_fallback: bool = False,
        read_to_file: str | None = None,
    ) -> DataFrame | bytes:
        """Download training data for this connection. Data Storage."""

        if self._api_client is not None:
            # note: as we need to load a data into the memory,
            # we are using pure requests and helpers from the API client
            asset_id = self.location.href.split("?")[0].split("/")[-1]

            # note: download data asset details
            asset_response = requests.get(
                self._api_client._href_definitions.get_data_asset_href(asset_id),
                params=self._api_client._params(),
                headers=self._api_client._get_headers(),
            )

            asset_details = self._api_client.data_assets._handle_response(
                200, "get assets", asset_response
            )

            attachment_id = asset_details["attachments"][0]["id"]
            response = requests.get(
                self._api_client._href_definitions.get_attachment_href(
                    asset_id, attachment_id
                ),
                params=self._api_client.data_assets._client._params(),
                headers=self._api_client.data_assets._client._get_headers(),
            )

            if response.status_code == 200:
                try:
                    attachment_details = response.json()
                    if "url" in attachment_details:
                        file_asset_url = attachment_details["url"]
                        if not file_asset_url.startswith("http"):
                            file_asset_url = (
                                self._api_client.credentials.url + file_asset_url
                            )
                        csv_response = requests.get(file_asset_url)

                        if csv_response.status_code != 200:
                            raise ApiRequestFailure(
                                "Failure during {}.".format("downloading model"),
                                csv_response,
                            )

                        downloaded_asset = csv_response.content

                        # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
                        buffer = io.BytesIO(downloaded_asset)

                        if is_flight_fallback:
                            _error_on_duplicate_columns_csv(
                                buffer,
                                self.auto_pipeline_params.get("csv_separator", ","),
                                self.auto_pipeline_params.get("encoding", "utf-8"),
                            )

                        if read_to_file:
                            with open(read_to_file, "wb") as file:
                                buffer.seek(0)
                                file.write(buffer.getvalue())

                        if binary:
                            buffer.seek(0)
                            return buffer.read()

                        data = try_load_dataset(
                            buffer=buffer,
                            sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                            separator=self.auto_pipeline_params.get(
                                "csv_separator", ","
                            ),
                            encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
                        )

                        return data

                except requests.exceptions.MissingSchema:
                    pass  # go to 'handle' part and check if we are able to download data asset from WS

            # note: read the csv url
            if "handle" in asset_details["attachments"][0]:
                attachment_url = asset_details["attachments"][0]["handle"]["key"]

                # note: make the whole url pointing out the csv
                artifact_content_url = (
                    f"{self._api_client._href_definitions.get_wsd_model_attachment_href()}"
                    f"{attachment_url}"
                )

                # note: stream the whole CSV file
                csv_response = requests.get(
                    artifact_content_url,
                    params=self._api_client._params(),
                    headers=self._api_client._get_headers(),
                    stream=True,
                )

                if csv_response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading model"), csv_response
                    )

                downloaded_asset = csv_response.content

                # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
                buffer = io.BytesIO(downloaded_asset)

                if is_flight_fallback:
                    _error_on_duplicate_columns_csv(
                        buffer,
                        self.auto_pipeline_params.get("csv_separator", ","),
                        self.auto_pipeline_params.get("encoding", "utf-8"),
                    )

                if binary:
                    buffer.seek(0)
                    return buffer.read()

                data = try_load_dataset(
                    buffer=buffer,
                    sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                    separator=self.auto_pipeline_params.get("csv_separator", ","),
                    encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
                )

                return data

            else:
                # NFS scenario
                connection_id = asset_details["attachments"][0]["connection_id"]
                connection_path = asset_details["attachments"][0]["connection_path"]

                return self._download_data_from_nfs_connection_using_id_and_path(
                    connection_id, connection_path, binary, is_flight_fallback
                )

        else:
            try:
                from ibm_watson_studio_lib import access_project_or_space

                wslib = access_project_or_space()
            except ModuleNotFoundError:
                raise NotImplementedError(
                    "This functionality can be run only on Watson Studio."
                )

            asset_id = self.location.href.split("?")[0].split("/")[-1]
            assets_list = wslib.assets.list_assets("asset")

            data_asset_name = None
            for asset in assets_list:
                if asset["asset_id"] == asset_id:
                    data_asset_name = asset["name"]

            if data_asset_name is None:
                raise FileNotFoundError("Cannot find data asset with id: {asset_id}")

            buffer = wslib.load_data(data_asset_name)

            if is_flight_fallback:
                _error_on_duplicate_columns_csv(
                    buffer,
                    self.auto_pipeline_params.get("csv_separator", ","),
                    self.auto_pipeline_params.get("encoding", "utf-8"),
                )

            if binary:
                buffer.seek(0)
                return buffer.read()

            data = try_load_dataset(
                buffer=buffer,
                sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                separator=self.auto_pipeline_params.get("csv_separator", ","),
                encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
            )

            return data

    def _download_training_data_from_file_system(
        self, binary: bool = False
    ) -> Tuple["DataFrame", bytes]:
        """Download training data for this connection. File system version."""

        try:
            url = (
                self._api_client._href_definitions.get_wsd_model_attachment_href()
                + f"/{self.location.path.split('/assets/', maxsplit=1)[-1]}"
            )
            # note: stream the whole CSV file
            csv_response = requests.get(
                url,
                params=self._api_client._params(),
                headers=self._api_client._get_headers(),
                stream=True,
            )

            if csv_response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading model"), csv_response
                )

            downloaded_asset = csv_response.content

            if binary:
                return downloaded_asset

            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(
                buffer=buffer,
                sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                separator=self.auto_pipeline_params.get("csv_separator", ","),
                encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
            )
        except (ApiRequestFailure, AttributeError):
            with open(self.location.path, "rb") as data_buffer:
                if binary:
                    return data_buffer.read()

                data = try_load_dataset(
                    buffer=data_buffer,
                    sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                    separator=self.auto_pipeline_params.get("csv_separator", ","),
                    encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
                )

        return data

    def _download_indices_from_file_system(self, location_path: str) -> "DataFrame":
        """Download indices for this connection. File system version."""

        try:
            url = (
                self._api_client._href_definitions.get_wsd_model_attachment_href()
                + f"/{location_path.split('/assets/', maxsplit=1)[-1]}"
            )
            # note: stream the whole CSV file
            csv_response = requests.get(
                url,
                params=self._api_client._params(),
                headers=self._api_client._get_headers(),
                stream=True,
            )

            if csv_response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading model"), csv_response
                )

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(
                buffer=buffer,
                sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
                separator=self.auto_pipeline_params.get("csv_separator", ","),
                encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
            )
        except (ApiRequestFailure, AttributeError):
            with open(location_path, "rb") as data_buffer:
                data = try_load_tar_gz(
                    buffer=data_buffer,
                    separator=self.auto_pipeline_params.get("csv_separator", ","),
                    encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
                )

        return data

    def _download_data_from_nfs_connection(
        self, binary: bool = False
    ) -> DataFrame | bytes:
        """Download training data for this connection. NFS."""

        # note: as we need to load a data into the memory,
        # we are using pure requests and helpers from the API client
        data_path = self.location.path
        connection_id = self.connection.asset_id

        return self._download_data_from_nfs_connection_using_id_and_path(
            connection_id, data_path, binary
        )

    def _download_data_from_nfs_connection_using_id_and_path(
        self,
        connection_id,
        connection_path,
        binary: bool = False,
        is_flight_fallback: bool = False,
    ) -> DataFrame | bytes:
        """Download training data for this connection. NFS."""

        # it means that it is on ICP env and it is before fit, so let's throw error
        if not self._api_client:
            raise CannotReadSavedRemoteDataBeforeFit()

        buffer = None
        # Note: workaround with volumes API as connections API changes data format
        try:
            connection_details = self._api_client.connections.get_details(connection_id)
        except ApiRequestFailure as conn_error:
            if os.environ.get("TRAINING_NFS_PATH"):
                # Note: Only viable on AutoAI runtime
                base_path = os.environ.get("TRAINING_NFS_PATH") + "/0"
                if connection_path.startswith("/"):
                    data_path = f"{base_path}{connection_path}"
                else:
                    data_path = f"{base_path}/{connection_path}"
                with open(data_path, "rb") as f:
                    buffer = io.BytesIO(f.read())
            else:
                raise conn_error

        if buffer is None:
            href = (
                self._api_client.volumes._client._href_definitions.volume_upload_href(
                    connection_details["entity"]["properties"]["volume"]
                )
            )
            full_href = f"{href[:-1]}{connection_path}"

            csv_response = requests.get(
                full_href, headers=self._api_client._get_headers(), stream=True
            )

            # Note: if file is written in directory we need to create different href for download
            if csv_response.status_code != 200:
                path_parts = connection_path.split("/")
                full_href = f"{href[:-1]}{'/'.join(path_parts[:-1])}%2F{path_parts[-1]}"

                csv_response = requests.get(
                    full_href, headers=self._api_client._get_headers(), stream=True
                )

                if csv_response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading data"), csv_response
                    )

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)

        if is_flight_fallback:
            _error_on_duplicate_columns_csv(
                buffer,
                self.auto_pipeline_params.get("csv_separator", ","),
                self.auto_pipeline_params.get("encoding", "utf-8"),
            )

        if binary:
            buffer.seek(0)
            return buffer.read()

        data = try_load_dataset(
            buffer=buffer,
            sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
            separator=self.auto_pipeline_params.get("csv_separator", ","),
            encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
        )

        return data

    def _download_data_from_cos(
        self, cos_client: "resource", binary: bool = False
    ) -> Union["DataFrame", bytes]:
        """Download training data for this connection. COS version"""
        location = (
            self.location.file_name
            if hasattr(self.location, "file_name")
            else self.location.path
        )
        try:
            file = cos_client.Object(self.location.bucket, location).get()
        except Exception as e:
            cos_objects_list = list(
                cos_client.Bucket(self.location.bucket).objects.filter(Prefix=location)
            )
            if len(cos_objects_list) > 0:
                file = list(
                    cos_client.Bucket(self.location.bucket).objects.filter(
                        Prefix=location
                    )
                )[0].get()
            else:
                raise e

        buffer = io.BytesIO(file["Body"].read())
        if binary:
            try:
                buffer.seek(0)
                return buffer.read()

            except Exception as e:
                raise Exception(f"Unable to read binary data from COS. Error: {e}")

        data = try_load_dataset(
            buffer=buffer,
            sheet_name=self.auto_pipeline_params.get("excel_sheet", 0),
            separator=self.auto_pipeline_params.get("csv_separator", ","),
            encoding=self.auto_pipeline_params.get("encoding", "utf-8"),
        )

        return data

    def _create_conn_details_for_container(self) -> dict:
        if self._api_client is not None:
            self._api_client._check_if_either_is_set()  # Space or project is required to use Container.
            if self._api_client.default_space_id is not None:
                if self._api_client.ICP_PLATFORM_SPACES:
                    raise ContainerTypeNotSupported()
                details = (
                    self._api_client.spaces._get_details(  # get details with TTL cache
                        self._api_client.default_space_id,
                        include="everything,credentials",
                    )
                )
            else:
                details = self._api_client.projects._get_details(  # get details with TTL cache
                    self._api_client.default_project_id,
                    include="everything,credentials",
                )

            match details["entity"]["storage"]["type"]:
                case "bmcos_object_storage":
                    properties = details["entity"]["storage"]["properties"]
                    creds = details["entity"]["storage"]["properties"][
                        "credentials"
                    ].get("admin")

                    if (
                        creds
                        and creds.get("access_key_id", False)
                        and creds.get("secret_access_key", False)
                    ):
                        pass
                    else:  # missing admin credentials
                        creds = details["entity"]["storage"]["properties"][
                            "credentials"
                        ].get("editor")

                    properties.update(creds)
                    properties["url"] = properties["endpoint_url"]
                    properties["access_key"] = properties.get("access_key_id")
                    properties["secret_key"] = properties.get("secret_access_key")

                    connection_details = {
                        "entity": {
                            "datasource_type": "bluemixcloudobjectstorage",
                            "properties": properties,
                        }
                    }
                case "amazon_s3":
                    properties = details["entity"]["storage"]["properties"]
                    properties["url"] = (
                        f"https://s3.{properties['bucket_region']}.amazonaws.com"
                    )
                    connection_details = {
                        "entity": {
                            "datasource_type": "amazons3",
                            "properties": properties,
                        }
                    }

                case _:
                    raise ValueError(
                        f"Container type not supported in the project with storage {details['entity']['storage']['type']}."
                    )

        else:
            try:
                from ibm_watson_studio_lib import access_project_or_space

                token = self._get_token_from_environment()

                if token is None:
                    raise NotImplementedError(
                        """To succesfully read the training data used in AutoAI experiment, you need to provide the project token.
                    **To insert the project token to your notebook:**
                        Click the More icon on your notebook toolbar and then click Insert project token.
                        Run the inserted code cell.
                    Note:
                    If you are told in a message that no project token exists, click the link in the message to be redirected to the project's Settings page where you can create a project token.
                    **To create a project token:**
                        Click New token in the Access tokens section on the Settings page of your project.
                        Enter a name, select Editor role for the project, and create a token.
                        Go back to your notebook, click the More icon on the notebook toolbar and then click Insert project token.
                        Run the inserted code cell."""
                    )

                token = token.split("Bearer ")[-1]

                wslib = access_project_or_space({"token": token})
                details = wslib.here.get_storage()

                properties = details["properties"]
                properties.update(properties["credentials"]["editor"])
                properties["url"] = properties["endpoint_url"]
                properties["access_key"] = properties.get("access_key_id")
                properties["secret_key"] = properties.get("secret_access_key")

                connection_details = {
                    "entity": {
                        "datasource_type": "bluemixcloudobjectstorage",
                        "properties": properties,
                    }
                }

            except ModuleNotFoundError:
                raise NotImplementedError(
                    "This functionality can be run only on Watson Studio."
                )

        return connection_details

    def _download_data_from_flight_service(
        self,
        data_location: "DataConnection",
        binary: bool = False,
        read_to_file: str = None,
        flight_parameters: dict = None,
        headers: dict = None,
        number_of_batch_rows: int = None,
        sampling_type: str = DEFAULT_SAMPLING_TYPE,
        return_data_as_iterator: bool = False,
        enable_sampling: bool = True,
        _return_subsampling_stats: bool = False,
        total_size_limit=DEFAULT_SAMPLE_SIZE_LIMIT,
        total_nrows_limit=None,
        total_percentage_limit=1.0,
    ):
        is_lib_installed(lib_name="pyarrow", minimum_version="3.0.0", install=True)

        # import the class only if flight scenario is enabled - do not import it in main import section
        from ibm_watsonx_ai.data_loaders.datasets.experiment import (
            TabularIterableDataset,
        )
        from ibm_watsonx_ai.data_loaders.experiment import ExperimentDataLoader

        if flight_parameters is None:
            flight_parameters = {"num_partitions": 4}

        dict_connection = data_location._to_dict()

        experiment_metadata = {
            "n_parallel_data_connections": flight_parameters.get("num_partitions", 4),
            "prediction_column": self.auto_pipeline_params.get("prediction_column"),
            "prediction_type": self.auto_pipeline_params.get("prediction_type"),
            "project_id": self._api_client.default_project_id,
            "space_id": self._api_client.default_space_id,
            "headers": self._api_client._get_headers() if headers is None else headers,
        }

        experiment_metadata.update(self.auto_pipeline_params)

        experiment_iterable_dataset_setup_parameters = dict(
            connection=dict_connection,
            enable_sampling=(
                enable_sampling if not binary else False
            ),  # don't use sampling to read binary
            experiment_metadata=experiment_metadata,
            binary_data=binary,
            read_to_file=read_to_file,
            flight_parameters=(
                flight_parameters if flight_parameters is not None else {}
            ),
            fallback_to_one_connection=False,
            _return_subsampling_stats=_return_subsampling_stats,
            number_of_batch_rows=number_of_batch_rows,
            sampling_type=sampling_type,
            _api_client=self._api_client,
            total_size_limit=total_size_limit,
            total_nrows_limit=total_nrows_limit,
            total_percentage_limit=total_percentage_limit,
        )

        iterable_dataset = TabularIterableDataset(
            **experiment_iterable_dataset_setup_parameters
        )
        data_loader = ExperimentDataLoader(dataset=iterable_dataset)

        try:
            if not return_data_as_iterator:
                for data in data_loader:
                    return data
            else:
                return data_loader
        except TypeError as e1:
            # note: retry if there is problem with types:
            # try download data with  infer_as_varchar set to 'true' if some error occurs
            # final data downloaded and converted to proper types might has smaller size than sample_size_limit,
            # because data limit is calculated on data downloaded as varchar, before the conversion to more optimal type.
            try:
                iterable_dataset = TabularIterableDataset(
                    **experiment_iterable_dataset_setup_parameters,
                    infer_as_varchar="true",
                )

                data_loader = ExperimentDataLoader(dataset=iterable_dataset)

                if not return_data_as_iterator:
                    for data in data_loader:
                        return data
                else:
                    return data_loader
            except Exception as e2:
                raise DataStreamError(
                    f"First attempt of downloading data failed with error: {e1}. \n"
                    f"Retry and use infer as varchar also failed with error: {e2}."
                )

    def _upload_data_via_flight_service(
        self,
        data_location: "DataConnection",
        data: DataFrame = None,
        file_path: str = None,
        remote_name: str = None,
        flight_parameters: dict = None,
        headers: dict = None,
        binary: bool = False,
    ):
        is_lib_installed(lib_name="pyarrow", minimum_version="3.0.0", install=True)

        # import the class only if flight scenario is enabled - do not import it in main import section
        from ibm_watsonx_ai.data_loaders.datasets.experiment import (
            TabularIterableDataset,
        )
        from ibm_watsonx_ai.helpers.connections import (
            DatabaseLocation,
            RemoteFileStorageLocation,
        )

        if flight_parameters is None:
            flight_parameters = {"num_partitions": 1}

        dict_connection = data_location._to_dict()

        if remote_name:
            dict_connection["location"]["path"] = self._get_path_with_remote_name(
                dict_connection, remote_name
            )
        elif get_from_json(dict_connection, ["location", "file_name"]):
            dict_connection["location"]["path"] = dict_connection["location"][
                "file_name"
            ]

        experiment_metadata = {
            "n_parallel_data_connections": flight_parameters.get("num_partitions", 1),
            "project_id": self._api_client.default_project_id,
            "space_id": self._api_client.default_space_id,
            "headers": self._api_client._get_headers() if headers is None else headers,
        }

        experiment_metadata.update(self.auto_pipeline_params)
        if binary and isinstance(self.location, DatabaseLocation):
            write_mode = self._api_client.connections.get_write_mode_by_datasource_type(
                self._datasource_type
            )
            extra_interaction_properties = {"write_mode": write_mode}
        elif isinstance(self.location, RemoteFileStorageLocation) and getattr(
            self.location, "container", None
        ):
            extra_interaction_properties = {
                "blob_type": "block",
            }
        else:
            extra_interaction_properties = {}

        iterable_dataset = TabularIterableDataset(
            connection=dict_connection,
            experiment_metadata=experiment_metadata,
            binary_data=True if file_path is not None else False,
            flight_parameters=(
                flight_parameters if flight_parameters is not None else {}
            ),
            _api_client=self._api_client,
            extra_interaction_properties=extra_interaction_properties,
        )

        if data is not None:
            try:
                iterable_dataset.write(data=data)

            except Exception as e:
                if "gRPC message exceeds maximum size" in str(e):
                    raise ValueError(
                        f"Exceeds maximum data size. Please provide data file path "
                        f"instead of the pandas DataFrame to upload data in binary mode. Error: {e}"
                    )

                else:
                    raise e

        else:
            if isinstance(self.location, DatabaseLocation) and not binary:
                if ".csv" in file_path:
                    df = pd.read_csv(file_path)
                    iterable_dataset.write(data=df)
                else:
                    raise ValueError(
                        f"Cannot upload file {file_path} to database. Please provide binary=True "
                        f"flag or provide file path to CSV file or provide pandas DataFrame. "
                    )
            else:
                iterable_dataset.write(file_path=file_path)

    def _upload_data_to_file_system(
        self, location: str, data: io.BufferedReader, remote_name: str | None = None
    ) -> None:
        filename, ext = os.path.splitext(os.path.basename(location))
        if filename == "." or ext.endswith("."):
            raise ValueError(
                f"The provided `location.path`: '{location}' is invalid to upload data."
            )
        elif ext:
            is_filename_location = True
        else:
            is_filename_location = False

        if remote_name:
            if location:
                if is_filename_location:
                    location = os.path.dirname(location)
                asset_path = (
                    location.split("/assets/", maxsplit=1)[-1] + "/" + remote_name
                )
            else:
                asset_path = "/" + remote_name
        elif is_filename_location:
            asset_path = "/" + location
        elif data.raw.name:
            asset_path = "/" + data.raw.name
        else:
            raise ValueError(
                "The `remote_name` and `location` fields provided are invalid."
            )

        if "//" in asset_path:
            asset_path = asset_path.replace("//", "/")

        content_upload_url = (
            self._api_client._href_definitions.get_wsd_model_attachment_href()
            + asset_path
        )

        response = requests.put(
            content_upload_url,
            files={
                "file": (
                    "native",
                    data,
                    "application/octet-stream",
                    {"Expires": "0"},
                )
            },
            headers=self._api_client._get_headers(no_content_type=True),
            params=self._api_client._params(),
        )

        if response.status_code == 200:
            self._api_client.repository._handle_response(
                200,
                "uploading asset to file system",
                response,
                _silent_response_logging=True,
            )
        else:
            self._api_client.repository._handle_response(
                201,
                "uploading asset to file system",
                response,
                _silent_response_logging=True,
            )

    @staticmethod
    def _get_path_with_remote_name(dict_connection: dict, remote_name: str) -> str:
        if get_from_json(dict_connection, ["location", "path"]):
            updated_path = dict_connection["location"]["path"] + "/" + remote_name
            updated_path = updated_path.replace("//", "/")
        elif get_from_json(dict_connection, ["location", "file_name"]):
            actual_path = dict_connection["location"]["file_name"]
            last_slash_index = actual_path.rfind("/") if "/" in actual_path else 0

            if "." in actual_path[last_slash_index:]:
                actual_path = actual_path[:-last_slash_index]

            if actual_path:
                updated_path = actual_path + "/" + remote_name
                updated_path = updated_path.replace("//", "/")
            else:
                updated_path = remote_name
        else:
            updated_path = remote_name

        return updated_path

    @staticmethod
    def _get_token_from_environment():
        if os.environ.get("RUNTIME_ENV_ACCESS_TOKEN_FILE"):
            with open(os.environ.get("RUNTIME_ENV_ACCESS_TOKEN_FILE"), "r") as f:
                token = f.read()
        else:
            token = os.environ.get("USER_ACCESS_TOKEN")

        if token:
            token = token.replace("Bearer ", "")

        return token

    def _is_size_acceptable(self):
        """
        Checks if data asset size is acceptable to download based on available memory in pod (MEM env variable)/ T-shirt size.
        Returns: True when data asset size is equal or lower than T-shirt limitation or when limitation is not set (outside autoai pod).
                False when data asset size is known and is above supported limit.
                None when data asset size is unknown or data_connection is not data asset type.
        """
        from ibm_watsonx_ai.utils.autoai.connection import get_max_sample_size_limit

        if self._api_client and self.type == "data_asset":
            items = (
                self.connection.href.split("/")
                if self.connection is not None and hasattr(self.connection, "href")
                else self.location.href.split("/")
            )
            data_asset_id = items[-1].split("?")[0]
            asset_size = get_from_json(
                self._api_client.data_assets.get_details(data_asset_id),
                ["metadata", "size"],
                0,
            )

            if asset_size == 0:  # data size unknown
                return None
            elif (
                asset_size <= get_max_sample_size_limit()
            ):  # data size is within acceptable range
                return True
            elif not os.environ.get("MEM", False):
                return True  # no limitation were set
            else:
                return False  # data size is above supported limit
        else:
            return None  # not a data asset
