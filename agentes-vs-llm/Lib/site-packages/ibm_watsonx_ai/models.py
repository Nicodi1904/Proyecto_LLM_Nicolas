#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import copy
import glob
import importlib.metadata
import json
import os
import shutil
import tarfile
import time
import uuid
import zipfile
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    BinaryIO,
    Generator,
    Literal,
    TypeAlias,
    cast,
    overload,
)
from warnings import catch_warnings, simplefilter, warn

import httpx

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.href_definitions import (
    API_VERSION,
    LIBRARIES,
    PIPELINES,
    RUNTIMES,
    SPACES,
)
from ibm_watsonx_ai.libs.repo.mlrepository import MetaProps
from ibm_watsonx_ai.libs.repo.mlrepositoryartifact import MLRepositoryArtifact
from ibm_watsonx_ai.libs.repo.util.compression_util import CompressionUtil
from ibm_watsonx_ai.libs.repo.util.unique_id_gen import uid_generate
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import LibraryMetaNames, ModelMetaNames
from ibm_watsonx_ai.utils import (
    MODEL_DETAILS_TYPE,
    is_lale_pipeline,
    load_model_from_directory,
)
from ibm_watsonx_ai.utils.autoai.utils import (
    adownload_request_json,
    aload_file_from_file_system_nonautoai,
    aprepare_auto_ai_model_to_publish,
    aprepare_auto_ai_model_to_publish_normal_scenario,
    check_if_ts_pipeline_is_winner,
    download_request_json,
    get_autoai_run_id_from_experiment_metadata,
    init_cos_client,
    load_file_from_file_system_nonautoai,
    prepare_auto_ai_model_to_publish,
    prepare_auto_ai_model_to_publish_normal_scenario,
)
from ibm_watsonx_ai.utils.deployment.errors import ModelPromotionFailed, PromotionFailed
from ibm_watsonx_ai.utils.utils import (
    AsyncFileReader,
    _get_id_from_deprecated_uid,
    get_from_json,
)
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    UnexpectedType,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import numpy
    import pandas
    import pyspark.ml.pipeline
    import pyspark.sql
    import requests as Requests

    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.sw_spec import SpecStates

    LabelColumnNamesType: TypeAlias = (
        numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str]
    )
    TrainingDataType: TypeAlias = (
        pandas.DataFrame | numpy.ndarray | pyspark.sql.DataFrame | list
    )
    TrainingTargetType: TypeAlias = (
        pandas.DataFrame | pandas.Series | numpy.ndarray | list
    )
    FeatureNamesArrayType: TypeAlias = numpy.ndarray | list

PipelineType: TypeAlias = Any
MLModelType: TypeAlias = Any


class Models(WMLResource):
    """Store and manage models."""

    ConfigurationMetaNames = ModelMetaNames()
    """MetaNames for models creation."""

    LibraryMetaNames = LibraryMetaNames()
    """MetaNames for libraries creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

        if self._client.ICP_PLATFORM_SPACES:
            self.default_space_id = client.default_space_id

    def _save_library_archive(self, ml_pipeline: "pyspark.ml.pipeline.Pipeline") -> str:
        gen_id = uid_generate(20)
        temp_dir_name = f"library{gen_id}"

        temp_dir = os.path.join(".", temp_dir_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        ml_pipeline.write().overwrite().save(temp_dir)
        archive_path = self._compress_artifact(temp_dir)
        shutil.rmtree(temp_dir)

        return archive_path

    def _compress_artifact(self, compress_artifact: str) -> str:
        tar_filename = "library_content.tar"
        gz_filename = f"{tar_filename}.gz"

        CompressionUtil.create_tar(compress_artifact, ".", tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)

        return gz_filename

    def _create_pipeline_input(
        self, lib_href: str, name: str, space_id: str | None = None
    ) -> dict[str, Any]:
        metadata = {
            self._client.pipelines.ConfigurationMetaNames.NAME: f"{name}_{uid_generate(8)}",
            self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "dlaas_only",
                "pipelines": [
                    {
                        "id": "dlaas_only",
                        "runtime_ref": "spark",
                        "nodes": [
                            {
                                "id": "repository",
                                "type": "model_node",
                                "inputs": [],
                                "outputs": [],
                                "parameters": {
                                    "training_lib_href": (
                                        lib_href
                                        if self._client.ICP_PLATFORM_SPACES
                                        else f"{lib_href}/content"
                                    )
                                },
                            }
                        ],
                    }
                ],
            },
        }

        if space_id is not None:
            metadata[self._client.pipelines.ConfigurationMetaNames.SPACE_ID] = space_id

        if self._client.default_project_id is not None:
            metadata["project"] = {
                "href": f"/v2/projects/{self._client.default_project_id}"
            }

        return metadata

    def _download_model_content(self, model_id: str, artifact_url: str) -> bytes | Any:
        params = self._client._params()
        params["content_format"] = "native"

        response = requests.get(
            self._client._href_definitions.get_model_download_href(model_id),
            params=params,
            headers=self._client._get_headers(),
            stream=True,
        )

        if response.status_code != 200:
            raise ApiRequestFailure("Failure during downloading model.", response)

        self._logger.info(
            "Successfully downloaded artifact with artifact_url: %s", artifact_url
        )

        return response.content

    async def _adownload_model_content(
        self, model_id: str, artifact_url: str
    ) -> bytes | Any:
        params = self._client._params()
        params["content_format"] = "native"

        async with self._client.async_httpx_client.stream(
            method="GET",
            url=self._client._href_definitions.get_model_download_href(model_id),
            params=params,
            headers=await self._client._aget_headers(),
        ) as response:
            if response.status_code != 200:
                raise ApiRequestFailure("Failure during downloading model.", response)

            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s", artifact_url
            )

            return await response.aread()

    @staticmethod
    def _save_tar_gz_into_temporary_directory(
        tar_gz_content: bytes, artifact_url: str
    ) -> str:
        try:
            gen_id = uid_generate(20)
            temp_dir_name = f"hdfs{gen_id}"

            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            file_name = temp_dir + "/download.tar.gz"
            with open(file_name, "wb") as f:
                f.write(tar_gz_content)

            return file_name
        except IOError as e:
            raise WMLClientError(
                f"Saving model with artifact_url: '{artifact_url}' failed.", e
            )

    def _create_tf_model_instance_from_archive(
        self, tar_gz_archive_path: str, artifact_url: str
    ) -> Any:
        tar_archive_path = tar_gz_archive_path[: -len(".gz")]
        directory_path = os.path.dirname(tar_gz_archive_path)

        try:
            CompressionUtil.decompress_file_gzip(
                gzip_filepath=tar_gz_archive_path, filepath=tar_archive_path
            )
            CompressionUtil.extract_tar(tar_archive_path, directory_path)
            os.remove(tar_archive_path)

            import tensorflow as tf  # pylint: disable=import-outside-toplevel

            if glob.glob(directory_path + "/sequential_model.h5"):
                model_instance = tf.keras.models.load_model(
                    directory_path + "/sequential_model.h5",
                    custom_objects=None,
                    compile=True,
                )
            elif glob.glob(directory_path + "/saved_model.pb"):
                model_instance = tf.saved_model.load(directory_path)
            else:
                raise WMLClientError(
                    f"Load model with artifact_url: '{artifact_url}' failed."
                )

            return model_instance
        except IOError as e:
            raise WMLClientError(
                f"Saving model with artifact_url: '{artifact_url}' failed.", e
            )

    def _tf2x_load_model_instance(self, model_id: str) -> Any:
        artifact_url = self._client._href_definitions.get_model_last_version_href(
            model_id
        )

        downloaded_model = self._download_model_content(model_id, artifact_url)

        tar_gz_archive_path = self._save_tar_gz_into_temporary_directory(
            downloaded_model, artifact_url
        )

        return self._create_tf_model_instance_from_archive(
            tar_gz_archive_path, artifact_url
        )

    async def _atf2x_load_model_instance(self, model_id: str) -> Any:
        artifact_url = self._client._href_definitions.get_model_last_version_href(
            model_id
        )

        downloaded_model = await self._adownload_model_content(model_id, artifact_url)

        tar_gz_archive_path = self._save_tar_gz_into_temporary_directory(
            downloaded_model, artifact_url
        )

        return self._create_tf_model_instance_from_archive(
            tar_gz_archive_path, artifact_url
        )

    def _get_pyspark_pipeline_href(
        self, name: str, pipeline: "pyspark.ml.pipeline.Pipeline"
    ) -> str:
        model_definition_props = {
            self._client.model_definitions.ConfigurationMetaNames.NAME: f"{name}_{uid_generate(8)}",
            self._client.model_definitions.ConfigurationMetaNames.VERSION: "1.0",
            self._client.model_definitions.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "versions": ["3.6"],
            },
        }

        library_tar = self._save_library_archive(pipeline)

        training_library = self._client.model_definitions.store(
            library_tar, model_definition_props
        )

        library_href = self._client.model_definitions.get_href(training_library)
        library_href = library_href.split("?", 1)[0]  # temp fix for stripping space_id

        pipeline_metadata = self._create_pipeline_input(
            library_href, name, space_id=None
        )

        pipeline_details = self._client.pipelines.store(pipeline_metadata)
        return self._client.pipelines.get_href(pipeline_details)

    async def _aget_pyspark_pipeline_href(
        self, name: str, pipeline: "pyspark.ml.pipeline.Pipeline"
    ) -> str:
        model_definition_props = {
            self._client.model_definitions.ConfigurationMetaNames.NAME: f"{name}_{uid_generate(8)}",
            self._client.model_definitions.ConfigurationMetaNames.VERSION: "1.0",
            self._client.model_definitions.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "versions": ["3.6"],
            },
        }

        library_tar = self._save_library_archive(pipeline)

        training_library = await self._client.model_definitions.astore(
            library_tar, model_definition_props
        )

        library_href = self._client.model_definitions.get_href(training_library)
        library_href = library_href.split("?", 1)[0]  # temp fix for stripping space_id

        pipeline_metadata = self._create_pipeline_input(
            library_href, name, space_id=None
        )

        pipeline_details = await self._client.pipelines.astore(pipeline_metadata)
        return self._client.pipelines.get_href(pipeline_details)

    def _prepare_pyspark_ml_repository_artifact_meta_props(
        self, meta_props: dict[str, Any]
    ):
        if (
            self.ConfigurationMetaNames.SPACE_ID in meta_props
            and meta_props[self._client.repository.ModelMetaNames.SPACE_ID] is not None
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
            )
            meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                "href": (
                    f"{API_VERSION}{SPACES}/"
                    + meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                )
            }
        elif self._client.default_project_id is not None:
            meta_props["project"] = {
                "href": f"/v2/projects/{self._client.default_project_id}"
            }

        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.RUNTIME_ID, str, False
            )
            meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID] = {
                "href": (
                    f"{API_VERSION}{RUNTIMES}/"
                    + meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID]
                )
            }

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_ID)

        if self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props:
            self._validate_meta_prop(
                meta_props,
                self.ConfigurationMetaNames.TRAINING_LIB_ID,
                str,
                False,
            )
            meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_ID] = {
                "href": (
                    f"{API_VERSION}{LIBRARIES}/"
                    + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_ID]
                )
            }

        return meta_props

    def _get_pyspark_ml_repository_artifact_meta_props(
        self,
        meta_props: dict[str, Any],
        pipeline: PipelineType | None,
        training_data: TrainingDataType | None,
    ) -> MetaProps:
        if pipeline is None or training_data is None:
            raise WMLClientError(
                "pipeline and training_data are expected for spark models."
            )

        meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
            "href": self._get_pyspark_pipeline_href(
                meta_props[self.ConfigurationMetaNames.NAME], pipeline
            )
        }

        return MetaProps(
            self._prepare_pyspark_ml_repository_artifact_meta_props(meta_props)
        )

    async def _aget_pyspark_ml_repository_artifact_meta_props(
        self,
        meta_props: dict[str, Any],
        pipeline: PipelineType | None,
        training_data: TrainingDataType | None,
    ) -> MetaProps:
        if pipeline is None or training_data is None:
            raise WMLClientError(
                "pipeline and training_data are expected for spark models."
            )

        meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
            "href": await self._aget_pyspark_pipeline_href(
                meta_props[self.ConfigurationMetaNames.NAME], pipeline
            )
        }

        return MetaProps(
            self._prepare_pyspark_ml_repository_artifact_meta_props(meta_props)
        )

    def _get_object_ml_repository_artifact_meta_props(self, meta_props: dict[str, Any]):
        if (
            self.ConfigurationMetaNames.SPACE_ID in meta_props
            and meta_props[self._client.repository.ModelMetaNames.SPACE_ID] is not None
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
            )
            meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                "href": (
                    f"{API_VERSION}{SPACES}/"
                    + meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                )
            }

        if self._client.default_space_id is not None:
            meta_props["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            meta_props["project_id"] = self._client.default_project_id
        else:
            raise WMLClientError(
                "It is mandatory is set the space or project. "
                "Use client.set.default_space(<SPACE_ID>) to set the space "
                "or use client.set.default_project(<PROJECT_ID)"
            )

        if self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                    "href": f"{API_VERSION}{SPACES}/{self._client.default_space_id}"
                }
            else:
                meta_props["project"] = {
                    "href": f"/v2/projects/{self._client.default_project_id}"
                }

        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.RUNTIME_ID, str, False
            )
            meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID] = {
                "href": (
                    f"{API_VERSION}{RUNTIMES}/"
                    + meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID]
                )
            }

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            if self._client.CPD_version:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.SOFTWARE_SPEC_ID,
                    str,
                    False,
                )
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                }
            else:
                meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_ID)

        if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
            )
            meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                "href": (
                    f"{API_VERSION}{PIPELINES}/"
                    + meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
                )
            }

        if self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props:
            self._validate_meta_prop(
                meta_props,
                self.ConfigurationMetaNames.TRAINING_LIB_ID,
                str,
                False,
            )
            meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_ID] = {
                "href": (
                    f"{API_VERSION}{LIBRARIES}/"
                    + meta_props[self._client.repository.ModelMetaNames.TRAINING_LIB_ID]
                )
            }

        return MetaProps(meta_props)

    def _create_cloud_model_payload(
        self,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)
        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            metadata["project_id"] = self._client.default_project_id
        else:
            raise WMLClientError(
                "It is mandatory is set the space or Project. \
                Use client.set.default_space(<SPACE_ID>) to set the space or"
                " Use client.set.default_project(<PROJECT_ID)"
            )

        if (
            self.ConfigurationMetaNames.RUNTIME_ID in meta_props
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input.  RUNTIME_ID is not supported in cloud environment. Specify SOFTWARE_SPEC_ID"
            )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
            )
            metadata[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            }

        if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
            )
            metadata[self.ConfigurationMetaNames.PIPELINE_ID] = {
                "id": meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
            }

        if self.ConfigurationMetaNames.MODEL_DEFINITION_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_ID, str, False
            )
            metadata[self.ConfigurationMetaNames.MODEL_DEFINITION_ID] = {
                "id": meta_props[
                    self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                ]
            }

        if (
            self.ConfigurationMetaNames.IMPORT in meta_props
            and meta_props[self.ConfigurationMetaNames.IMPORT] is not None
        ):
            print(
                "WARNING: Invalid input. IMPORT is not supported in cloud environment."
            )

        if (
            self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props
            and meta_props[self.ConfigurationMetaNames.IMPORT] is not None
        ):
            print(
                "WARNING: Invalid input. TRAINING_LIB_ID is not supported in cloud environment."
            )

        input_schema = []
        if (
            self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None
        ):
            if isinstance(
                meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], list
            ):
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.INPUT_DATA_SCHEMA,
                    list,
                    False,
                )
                input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.INPUT_DATA_SCHEMA,
                    dict,
                    False,
                )
                input_schema = [
                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
                ]

            metadata.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMA)

        output_schema = []
        if (
            self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None
        ):
            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith("do-"):
                if isinstance(
                    meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA], dict
                ):
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                        dict,
                        False,
                    )
                    output_schema = [
                        meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                    ]
                else:
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                        list,
                        False,
                    )
                    output_schema = meta_props[
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA
                    ]
            else:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                    dict,
                    False,
                )
                output_schema = [
                    meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                ]
            metadata.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA)

        if input_schema or output_schema:
            metadata["schemas"] = {"input": input_schema, "output": output_schema}

        if label_column_names:
            metadata["label_column"] = label_column_names[0]

        return metadata

    def _publish_from_object(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""

        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        if (
            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
            and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. It is mandatory to provide RUNTIME_ID or "
                "SOFTWARE_SPEC_ID in meta_props. RUNTIME_ID is deprecated"
            )

        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            runtime_id_deprecated_warning = (
                "RUNTIME_ID is deprecated and will be removed in future. "
                "Instead, please use SOFTWARE_SPEC_ID."
            )
            warn(runtime_id_deprecated_warning, category=DeprecationWarning)

        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                ml_artifact_meta_props = (
                    self._get_pyspark_ml_repository_artifact_meta_props(
                        meta_props, pipeline, training_data
                    )
                )
            else:
                ml_artifact_meta_props = (
                    self._get_object_ml_repository_artifact_meta_props(meta_props)
                )

            model_artifact = MLRepositoryArtifact(
                model,
                name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                meta_props=ml_artifact_meta_props,
                training_data=training_data,
                training_target=training_target,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

            saved_model = self._client.repository._ml_repository_client.models.save(
                model_artifact,
                query_param=self._client._params()
                if self._client.ICP_PLATFORM_SPACES
                else None,
            )
        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)

        return self.get_details(saved_model.uid)

    async def _apublish_from_object(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud asynchronously."""

        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        if (
            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
            and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. It is mandatory to provide RUNTIME_ID or "
                "SOFTWARE_SPEC_ID in meta_props. RUNTIME_ID is deprecated"
            )

        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            runtime_id_deprecated_warning = (
                "RUNTIME_ID is deprecated and will be removed in future. "
                "Instead, please use SOFTWARE_SPEC_ID."
            )
            warn(runtime_id_deprecated_warning, category=DeprecationWarning)

        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                ml_artifact_meta_props = (
                    await self._aget_pyspark_ml_repository_artifact_meta_props(
                        meta_props, pipeline, training_data
                    )
                )
            else:
                ml_artifact_meta_props = (
                    self._get_object_ml_repository_artifact_meta_props(meta_props)
                )

            model_artifact = MLRepositoryArtifact(
                model,
                name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                meta_props=ml_artifact_meta_props,
                training_data=training_data,
                training_target=training_target,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

            saved_model = self._client.repository._ml_repository_client.models.save(
                model_artifact,
                query_param=self._client._params()
                if self._client.ICP_PLATFORM_SPACES
                else None,
            )
        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)

        return await self.aget_details(saved_model.uid)

    @staticmethod
    def _get_subtraining_details(
        training_details: dict[str, Any], subtrainingId: str
    ) -> dict[str, Any]:
        try:
            return next(
                subtraining_details
                for subtraining_details in training_details["resources"]
                if subtraining_details["metadata"]["guid"] == subtrainingId
            )
        except StopIteration:
            raise WMLClientError(f"Subtraining ID {subtrainingId} not found")

    def _get_subtraining_details_from_parent(
        self, parent_id: str, subtrainingId: str
    ) -> dict[str, Any]:
        response = requests.get(
            self._client._href_definitions.get_trainings_href(),
            params=self._client._params() | {"parent_id": parent_id},
            headers=self._client._get_headers(),
        )
        training_details = self._handle_response(200, "Get training details", response)

        return self._get_subtraining_details(training_details, subtrainingId)

    async def _aget_subtraining_details_from_parent(
        self, parent_id: str, subtrainingId: str
    ) -> dict[str, Any]:
        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_trainings_href(),
            params=self._client._params() | {"parent_id": parent_id},
            headers=await self._client._aget_headers(),
        )
        training_details = self._handle_response(200, "Get training details", response)

        return self._get_subtraining_details(training_details, subtrainingId)

    def _get_model_directory(
        self, training_details: dict, model_id: str, round_number: int | None
    ) -> str:
        training_state = get_from_json(training_details, ["entity", "status", "state"])
        if training_state in {"failed", "pending"}:
            raise WMLClientError(
                "Training is not successfully completed for the given training_id. "
                "Please check the status of training run. "
                "Training should be completed successfully to store the model."
            )

        model_dir = model_id
        if (
            "federated_learning" in training_details["entity"]
            and round_number is not None
        ):
            if not get_from_json(
                training_details, ["entity", "federated_learning", "rounds"], False
            ):
                raise WMLClientError(
                    "The Federated Learning experiment was not configured "
                    "to save intermediate models"
                )

            rounds = get_from_json(
                training_details, ["entity", "federated_learning", "rounds"], 0
            )

            if 0 < round_number < rounds:
                # intermediate models
                model_dir = model_dir + "_" + str(round_number)
            elif round_number != rounds:
                raise WMLClientError(
                    f"Invalid input. round_number should be an int between 1 and {rounds}"
                )

        return model_dir

    def _publish_from_training(
        self,
        model_id: str,
        subtrainingId: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""

        model_meta = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

        try:
            details = self._client.training.get_details(model_id, _internal=True)
        except ApiRequestFailure:
            raise UnexpectedType(
                "model parameter", "model path / training_id", model_id
            )

        model_type = ""
        is_onnx = "onnx" in str(details) and "onnxruntime" in meta_props.get("type", "")

        subtraining_details = {}
        pipeline_id = ""

        # Check if the training is created from pipeline or experiment
        if "pipeline" in details["entity"]:
            pipeline_id = details["entity"]["pipeline"]["id"]
            if "model_type" in details["entity"]["pipeline"]:
                model_type = details["entity"]["pipeline"]["model_type"]

        if "experiment" in details["entity"]:
            subtraining_details = self._get_subtraining_details_from_parent(
                model_id, subtrainingId
            )
            model_meta["import"] = subtraining_details["entity"]["results_reference"]
            if "pipeline" in subtraining_details["entity"]:
                pipeline_id = subtraining_details["entity"]["pipeline"]["id"]
                if "model_type" in subtraining_details["entity"]["pipeline"]:
                    model_type = subtraining_details["entity"]["pipeline"]["model_type"]
        else:
            model_meta["import"] = details["entity"]["results_reference"]

        if (
            "experiment" in details["entity"]
            and "pipeline" in subtraining_details["entity"]
        ):
            definition_details = self._client.pipelines.get_details(pipeline_id)
            runtime = definition_details["entity"]["document"]["runtimes"][0]
            runtime_id = f"{runtime['name']}_{runtime['version'].split('-')[0]}-py3"

            if not model_type:
                model_type = f"{runtime['name']}_{runtime['version'].split('-')[0]}"

            if self.ConfigurationMetaNames.TYPE not in meta_props:
                model_meta["type"] = model_type

            if self.ConfigurationMetaNames.RUNTIME_ID not in meta_props:
                model_meta["runtime"] = {"href": "/v4/runtimes/" + runtime_id}
        elif "pipeline" in details["entity"]:
            definition_details = self._client.pipelines.get_details(pipeline_id)
            runtime = definition_details["entity"]["document"]["runtimes"][0]

            if not model_type:
                model_type = (
                    f"{runtime['name']}_{runtime.get('version', '0.1').split('-')[0]}"
                )

            if self.ConfigurationMetaNames.TYPE not in meta_props:
                model_meta["type"] = model_type

        if label_column_names:
            model_meta["label_column"] = label_column_names[0]

        model_dir = self._get_model_directory(details, model_id, round_number)

        asset_url = (
            details["entity"]["results_reference"]["location"]["assets_path"]
            + f"/{model_dir}/resources/wml_model/request.json"
        )
        if is_onnx:
            pass
        elif self._client.ICP_PLATFORM_SPACES:
            try:
                asset_parts = asset_url.split("/")
                asset_url = "/".join(asset_parts[asset_parts.index("assets") + 1 :])
                request_str = (
                    load_file_from_file_system_nonautoai(
                        api_client=self._client, file_path=asset_url
                    )
                    .read()
                    .decode()
                )

                if json.loads(request_str).get("code") == 404:
                    raise Exception("Not found file.")
            except Exception:
                asset_url = f"trainings/{model_id}/assets/{model_dir}/resources/wml_model/request.json"
                request_str = (
                    load_file_from_file_system_nonautoai(
                        api_client=self._client, file_path=asset_url
                    )
                    .read()
                    .decode()
                )
        else:
            if len(details["entity"]["results_reference"]["connection"]) > 1:
                cos_client = init_cos_client(
                    details["entity"]["results_reference"]["connection"]
                )
                bucket = details["entity"]["results_reference"]["location"]["bucket"]
            else:
                results_reference = DataConnection._from_dict(
                    details["entity"]["results_reference"]
                )
                results_reference.set_client(self._client)
                results_reference._check_if_connection_asset_is_s3()
                results_reference = results_reference._to_dict()

                cos_client = init_cos_client(results_reference["connection"])
                bucket = results_reference["location"].get(
                    "bucket", results_reference["connection"].get("bucket")
                )

            cos_client.meta.client.download_file(
                Bucket=bucket, Filename="request.json", Key=asset_url
            )
            with open("request.json", "r", encoding="utf-8") as f:
                request_str = f.read()

        if is_onnx:
            request_json: dict[str, dict] = copy.copy(meta_props)
        else:
            request_json: dict[str, dict] = json.loads(request_str)
        request_json["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        request_json["content_location"]["connection"] = details["entity"][
            "results_reference"
        ].get("connection", details["entity"]["results_reference"])

        if "space_id" in model_meta:
            request_json["space_id"] = model_meta["space_id"]
        else:
            request_json["project_id"] = model_meta["project_id"]

        if "label_column" in model_meta:
            request_json["label_column"] = model_meta["label_column"]
        if "pipeline" in request_json and not is_onnx:
            request_json.pop("pipeline")  # not needed for other space
        if "training_data_references" in request_json:
            request_json.pop("training_data_references")
        if "software_spec" in request_json:
            request_json.pop("software_spec")
            request_json["software_spec"] = {
                "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            }

        if is_onnx and isinstance(
            (val := request_json.get(self.ConfigurationMetaNames.PIPELINE_ID)), str
        ):
            request_json[self.ConfigurationMetaNames.PIPELINE_ID] = {"id": val}

        params = {}
        params["version"] = self._client.version_param
        creation_response = requests.post(
            self._client._href_definitions.get_published_models_href(),
            headers=self._client._get_headers(),
            json=request_json,
            params=params,
        )
        model_details = self._handle_response(
            202, "creating new model", creation_response
        )
        if is_onnx:
            return self._wait_for_content_import_completion(model_details)
        model_id = model_details["metadata"]["id"]
        return self.get_details(model_id)

    async def _apublish_from_training(
        self,
        model_id: str,
        subtrainingId: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud asynchronously."""

        model_meta = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

        try:
            details = await self._client.training.aget_details(model_id, _internal=True)
        except ApiRequestFailure:
            raise UnexpectedType(
                "model parameter", "model path / training_id", model_id
            )

        model_type = ""
        is_onnx = "onnx" in str(details) and "onnxruntime" in meta_props.get("type", "")

        subtraining_details = {}
        pipeline_id = ""

        # Check if the training is created from pipeline or experiment
        if "pipeline" in details["entity"]:
            pipeline_id = details["entity"]["pipeline"]["id"]
            if "model_type" in details["entity"]["pipeline"]:
                model_type = details["entity"]["pipeline"]["model_type"]

        if "experiment" in details["entity"]:
            subtraining_details = await self._aget_subtraining_details_from_parent(
                model_id, subtrainingId
            )
            model_meta["import"] = subtraining_details["entity"]["results_reference"]
            if "pipeline" in subtraining_details["entity"]:
                pipeline_id = subtraining_details["entity"]["pipeline"]["id"]
                if "model_type" in subtraining_details["entity"]["pipeline"]:
                    model_type = subtraining_details["entity"]["pipeline"]["model_type"]
        else:
            model_meta["import"] = details["entity"]["results_reference"]

        if (
            "experiment" in details["entity"]
            and "pipeline" in subtraining_details["entity"]
        ):
            definition_details = await self._client.pipelines.aget_details(pipeline_id)
            runtime = definition_details["entity"]["document"]["runtimes"][0]
            runtime_id = f"{runtime['name']}_{runtime['version'].split('-')[0]}-py3"

            if not model_type:
                model_type = f"{runtime['name']}_{runtime['version'].split('-')[0]}"

            if self.ConfigurationMetaNames.TYPE not in meta_props:
                model_meta["type"] = model_type

            if self.ConfigurationMetaNames.RUNTIME_ID not in meta_props:
                model_meta["runtime"] = {"href": "/v4/runtimes/" + runtime_id}
        elif "pipeline" in details["entity"]:
            definition_details = await self._client.pipelines.aget_details(pipeline_id)
            runtime = definition_details["entity"]["document"]["runtimes"][0]

            if not model_type:
                model_type = (
                    f"{runtime['name']}_{runtime.get('version', '0.1').split('-')[0]}"
                )

            if self.ConfigurationMetaNames.TYPE not in meta_props:
                model_meta["type"] = model_type

        if label_column_names:
            model_meta["label_column"] = label_column_names[0]

        model_dir = self._get_model_directory(details, model_id, round_number)

        asset_url = (
            details["entity"]["results_reference"]["location"]["assets_path"]
            + f"/{model_dir}/resources/wml_model/request.json"
        )

        if is_onnx:
            pass
        elif self._client.ICP_PLATFORM_SPACES:
            try:
                asset_parts = asset_url.split("/")
                asset_url = "/".join(asset_parts[asset_parts.index("assets") + 1 :])
                request_str = (
                    (
                        await aload_file_from_file_system_nonautoai(
                            api_client=self._client, file_path=asset_url
                        )
                    )
                    .read()
                    .decode()
                )

                if json.loads(request_str).get("code") == 404:
                    raise Exception("Not found file.")
            except Exception:
                asset_url = f"trainings/{model_id}/assets/{model_dir}/resources/wml_model/request.json"
                request_str = (
                    (
                        await aload_file_from_file_system_nonautoai(
                            api_client=self._client, file_path=asset_url
                        )
                    )
                    .read()
                    .decode()
                )
        else:
            if len(details["entity"]["results_reference"]["connection"]) > 1:
                cos_client = init_cos_client(
                    details["entity"]["results_reference"]["connection"]
                )
                bucket = details["entity"]["results_reference"]["location"]["bucket"]
            else:
                results_reference = DataConnection._from_dict(
                    details["entity"]["results_reference"]
                )
                results_reference.set_client(self._client)
                results_reference._check_if_connection_asset_is_s3()
                results_reference = results_reference._to_dict()

                cos_client = init_cos_client(results_reference["connection"])
                bucket = results_reference["location"].get(
                    "bucket", results_reference["connection"].get("bucket")
                )

            cos_client.meta.client.download_file(
                Bucket=bucket, Filename="request.json", Key=asset_url
            )
            with open("request.json", "r", encoding="utf-8") as f:
                request_str = f.read()

        if is_onnx:
            request_json: dict[str, dict] = copy.copy(meta_props)
        else:
            request_json: dict[str, dict] = json.loads(request_str)
        request_json["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        request_json["content_location"]["connection"] = details["entity"][
            "results_reference"
        ].get("connection", details["entity"]["results_reference"])

        if "space_id" in model_meta:
            request_json["space_id"] = model_meta["space_id"]
        else:
            request_json["project_id"] = model_meta["project_id"]

        if "label_column" in model_meta:
            request_json["label_column"] = model_meta["label_column"]
        if "pipeline" in request_json and not is_onnx:
            request_json.pop("pipeline")  # not needed for other space
        if "training_data_references" in request_json:
            request_json.pop("training_data_references")
        if "software_spec" in request_json:
            request_json.pop("software_spec")
            request_json["software_spec"] = {
                "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            }

        if is_onnx and isinstance(
            (val := request_json.get(self.ConfigurationMetaNames.PIPELINE_ID)), str
        ):
            request_json[self.ConfigurationMetaNames.PIPELINE_ID] = {"id": val}

        params = {"version": self._client.version_param}
        creation_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_published_models_href(),
            headers=await self._client._aget_headers(),
            json=request_json,
            params=params,
        )
        model_details = self._handle_response(
            202, "creating new model", creation_response
        )
        if is_onnx:
            return await self._await_for_content_import_completion(model_details)
        model_id = model_details["metadata"]["id"]
        return await self.aget_details(model_id)

    def _wait_for_content_import_completion(self, model_details: dict) -> dict:
        if "entity" not in model_details:
            return model_details

        model_id = model_details["metadata"]["id"]
        end_time = time.time() + 60

        while (
            model_details["entity"].get("content_import_state") == "running"
            and time.time() < end_time
        ):
            time.sleep(2)
            model_details = self.get_details(model_id)

        return model_details

    def _store_auto_ai_model(
        self,
        model_path: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""
        model_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, client=self._client
        )

        # For V4 cloud prepare the metadata
        if "autoai_sdk" in model_path:
            input_payload = meta_props
        else:
            input_payload = copy.deepcopy(
                self._create_cloud_model_payload(
                    model_meta,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            )

        if label_column_names:
            input_payload["label_column"] = label_column_names[0]

        params = {"version": self._client.version_param}

        creation_response = requests.post(
            self._client._href_definitions.get_published_models_href(),
            params=params,
            headers=self._client._get_headers(),
            json=input_payload,
        )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return self._wait_for_content_import_completion(model_details)

    async def _await_for_content_import_completion(self, model_details: dict) -> dict:
        if "entity" not in model_details:
            return model_details

        model_id = model_details["metadata"]["id"]
        end_time = time.time() + 60

        while (
            model_details["entity"].get("content_import_state") == "running"
            and time.time() < end_time
        ):
            await asyncio.sleep(2)
            model_details = await self.aget_details(model_id)

        return model_details

    async def _astore_auto_ai_model(
        self,
        model_path: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud asynchronously."""
        model_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, client=self._client
        )

        # For V4 cloud prepare the metadata
        if "autoai_sdk" in model_path:
            input_payload = meta_props
        else:
            input_payload = copy.deepcopy(
                self._create_cloud_model_payload(
                    model_meta,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            )

        if label_column_names:
            input_payload["label_column"] = label_column_names[0]

        params = {"version": self._client.version_param}

        creation_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_published_models_href(),
            params=params,
            headers=await self._client._aget_headers(),
            json=input_payload,
        )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return await self._await_for_content_import_completion(model_details)

    def _upload_autoai_model_content(
        self, file: str | BinaryIO, url: str, qparams: dict[str, Any]
    ) -> Requests.Response:
        node_ids = None

        with zipfile.ZipFile(file) as zip_file:
            # Get a list of all archived file names from the zip
            list_of_file_names = zip_file.namelist()
            t1 = zip_file.extract("pipeline-model.json")

            with open(t1, "rb") as f2:
                data = json.load(f2)
                # note: we can have multiple nodes (OBM + KB)
                node_ids = [
                    node.get("id")
                    for node in get_from_json(data, ["pipelines", 0, "nodes"])
                ]

            if node_ids is None:
                raise WMLClientError(
                    "Invalid pipline-model.json content file. There is no node id value found"
                )

            qparams["content_format"] = "native"
            qparams["name"] = "pipeline-model.json"

            if "pipeline_node_id" in qparams.keys():
                qparams.pop("pipeline_node_id")

            with open(t1, "rb") as f2:
                response = requests.put(
                    url,
                    data=f2,
                    params=qparams,
                    headers=self._client._get_headers(content_type="application/json"),
                )

            list_of_file_names.remove("pipeline-model.json")

            # note: the file order is important, should be OBM model first then KB model
            for file_name, node_id in zip(list_of_file_names, node_ids):
                if not file_name.endswith(".tar.gz") and not file_name.endswith(".zip"):
                    continue

                qparams["content_format"] = "pipeline-node"
                qparams["pipeline_node_id"] = node_id
                qparams["name"] = file_name
                t2 = zip_file.extract(file_name)

                with open(t2, "rb") as f1:
                    response = requests.put(
                        url,
                        data=f1,
                        params=qparams,
                        headers=self._client._get_headers(
                            content_type="application/octet-stream"
                        ),
                    )

        return response

    def _publish_from_archive(
        self,
        path_to_archive: str,
        meta_props: dict[str, Any],
        version: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        url = self._client._href_definitions.get_published_models_href()
        payload = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

        while True:
            response = requests.post(
                url,
                json=payload,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                self._process_sw_spec_error(
                    response, meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                )

            if (
                response.status_code == 400
                and "hybrid_pipeline_software_specs" in response.text
                and "hybrid_pipeline_software_specs" in payload
            ):
                payload.pop("hybrid_pipeline_software_specs")
                continue

            result = self._handle_response(201, "creating model", response)
            model_id = self._get_required_element_from_dict(
                result, "model_details", ["metadata", "id"]
            )
            break

        content_url = self._client._href_definitions.get_published_model_content_href(
            model_id
        )

        with open(path_to_archive, "rb") as file:
            qparams = self._client._params()
            if path_to_archive.endswith(".xml"):
                qparams["content_format"] = "coreML"
                response = requests.put(
                    content_url,
                    data=file,
                    params=qparams,
                    headers=self._client._get_headers(content_type="application/xml"),
                )
            else:
                qparams["content_format"] = "native"
                qparams["version"] = self._client.version_param
                model_type = meta_props[self.ConfigurationMetaNames.TYPE]

                # update the content path for the Auto-ai model.
                if model_type == "wml-hybrid_0.1":
                    response = self._upload_autoai_model_content(
                        file, content_url, qparams
                    )
                else:
                    response = requests.put(
                        content_url,
                        data=file,
                        params=qparams,
                        headers=self._client._get_headers(
                            content_type="application/octet-stream"
                        ),
                    )

            if response.status_code not in {200, 201}:
                self.delete(model_id)

            self._handle_response(201, "uploading model content", response, False)

            if version is True:
                return self.get_details(f"{artifactid}/versions/{model_id}")

            return self.get_details(model_id)

    async def _aupload_autoai_model_content(
        self, file: str | BinaryIO, url: str, qparams: dict[str, Any]
    ) -> httpx.Response:
        node_ids = None

        with zipfile.ZipFile(file) as zip_file:
            # Get a list of all archived file names from the zip
            list_of_file_names = zip_file.namelist()
            t1 = zip_file.extract("pipeline-model.json")

            with open(t1, "rb") as f2:
                data = json.load(f2)
                # note: we can have multiple nodes (OBM + KB)
                node_ids = [
                    node.get("id")
                    for node in get_from_json(data, ["pipelines", 0, "nodes"])
                ]

            if node_ids is None:
                raise WMLClientError(
                    "Invalid pipline-model.json content file. There is no node id value found"
                )

            qparams["content_format"] = "native"
            qparams["name"] = "pipeline-model.json"

            if "pipeline_node_id" in qparams.keys():
                qparams.pop("pipeline_node_id")

            response = await self._client.async_httpx_client.put(
                url,
                content=AsyncFileReader(t1),
                params=qparams,
                headers=await self._client._aget_headers(
                    content_type="application/json"
                ),
            )

            list_of_file_names.remove("pipeline-model.json")

            # note: the file order is important, should be OBM model first then KB model
            for file_name, node_id in zip(list_of_file_names, node_ids):
                if not file_name.endswith(".tar.gz") and not file_name.endswith(".zip"):
                    continue

                qparams["content_format"] = "pipeline-node"
                qparams["pipeline_node_id"] = node_id
                qparams["name"] = file_name
                t2 = zip_file.extract(file_name)

                response = await self._client.async_httpx_client.put(
                    url,
                    content=AsyncFileReader(t2),
                    params=qparams,
                    headers=await self._client._aget_headers(
                        content_type="application/octet-stream"
                    ),
                )

        return response

    async def _apublish_from_archive(
        self,
        path_to_archive: str,
        meta_props: dict[str, Any],
        version: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        url = self._client._href_definitions.get_published_models_href()
        payload = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

        while True:
            response = await self._client.async_httpx_client.post(
                url,
                json=payload,
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                await self._aprocess_sw_spec_error(
                    response, meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                )

            if (
                response.status_code == 400
                and "hybrid_pipeline_software_specs" in response.text
                and "hybrid_pipeline_software_specs" in payload
            ):
                payload.pop("hybrid_pipeline_software_specs")
                continue

            result = self._handle_response(201, "creating model", response)
            model_id = self._get_required_element_from_dict(
                result, "model_details", ["metadata", "id"]
            )
            break

        content_url = self._client._href_definitions.get_published_model_content_href(
            model_id
        )

        qparams = self._client._params()
        if path_to_archive.endswith(".xml"):
            qparams["content_format"] = "coreML"
            response = await self._client.async_httpx_client.put(
                content_url,
                content=AsyncFileReader(path_to_archive),
                params=qparams,
                headers=await self._client._aget_headers(
                    content_type="application/xml"
                ),
            )
        else:
            qparams["content_format"] = "native"
            qparams["version"] = self._client.version_param
            model_type = meta_props[self.ConfigurationMetaNames.TYPE]

            # update the content path for the Auto-ai model.
            if model_type == "wml-hybrid_0.1":
                response = await self._aupload_autoai_model_content(
                    path_to_archive, content_url, qparams
                )
            else:
                response = await self._client.async_httpx_client.put(
                    content_url,
                    content=AsyncFileReader(path_to_archive),
                    params=qparams,
                    headers=await self._client._aget_headers(
                        content_type="application/octet-stream"
                    ),
                )

        if response.status_code not in {200, 201}:
            await self.adelete(model_id)

        self._handle_response(201, "uploading model content", response, False)

        return await self.aget_details(
            f"{artifactid}/versions/{model_id}" if version else model_id
        )

    def _publish_from_tensorflow_dir(
        self,
        model: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None,
        label_column_names: LabelColumnNamesType | None,
    ) -> dict:
        # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
        if os.path.basename(model) == "":
            model = os.path.dirname(model)
        filename = os.path.basename(model) + ".tar.gz"
        current_dir = os.getcwd()
        os.chdir(model)
        target_path = os.path.dirname(model)

        with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
            tar.add(".")

        os.chdir(current_dir)
        model_filepath = os.path.join(target_path, filename)

        return self._publish_from_archive(
            model_filepath,
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

    async def _apublish_from_tensorflow_dir(
        self,
        model: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None,
        label_column_names: LabelColumnNamesType | None,
    ) -> dict:
        # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
        if os.path.basename(model) == "":
            model = os.path.dirname(model)
        filename = os.path.basename(model) + ".tar.gz"
        current_dir = os.getcwd()
        os.chdir(model)
        target_path = os.path.dirname(model)

        with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
            tar.add(".")

        os.chdir(current_dir)
        model_filepath = os.path.join(target_path, filename)

        return await self._apublish_from_archive(
            model_filepath,
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

    def _model_content_compress_artifact(
        self, type_name: str, compress_artifact: str
    ) -> str:
        tar_filename = "{}_content.tar".format(type_name)
        gz_filename = "{}.gz".format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, ".", tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _save_tensorflow_model_to_directory(
        self,
        model: Any,
        save_format: str | None,
        signature: Any,
        options: Any,
        include_optimizer: Any,
    ):
        import tensorflow as tf

        gen_id = uid_generate(20)

        if save_format == "tf" or (
            save_format is None
            and "tensorflow.python.keras.engine.training.Model" in str(type(model))
        ):
            temp_dir_name = f"pb{gen_id}"
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            tf.saved_model.save(model, temp_dir, signatures=signature, options=options)
        elif save_format == "h5" or (
            save_format is None
            and "tensorflow.python.keras.engine.sequential.Sequential"
            in str(type(model))
        ):
            temp_dir_name = f"hdfs{gen_id}"
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            if importlib.metadata.version("keras") >= "3.0.0":
                tf.keras.models.save_model(
                    model,
                    temp_dir + "/sequential_model.h5",
                    include_optimizer=include_optimizer,
                )
            else:
                tf.keras.models.save_model(
                    model,
                    temp_dir + "/sequential_model.h5",
                    include_optimizer=include_optimizer,
                    save_format="h5",
                    signatures=None,
                    options=options,
                )
        elif isinstance(model, str) and model.endswith(".h5"):
            temp_dir_name = f"hdfs{gen_id}"
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                import shutil

                os.makedirs(temp_dir)
                shutil.copy2(model, temp_dir)
        else:
            raise WMLClientError(
                "Saving the tensorflow model requires the model of either tf format or h5 format for Sequential model."
            )

        return temp_dir_name, temp_dir

    def _store_tf_model(
        self,
        model: Any,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        tf_meta = None
        options = None
        signature = None
        save_format = None
        include_optimizer = None

        if (
            "tf_model_params" in meta_props
            and meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS] is not None
        ):
            tf_meta = copy.deepcopy(
                meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS]
            )
            save_format = tf_meta.get("save_format")
            options = tf_meta.get("options")
            signature = tf_meta.get("signature")
            include_optimizer = tf_meta.get("include_optimizer")

        temp_dir_name, temp_dir = self._save_tensorflow_model_to_directory(
            model, save_format, signature, options, include_optimizer
        )

        path_to_archive = self._model_content_compress_artifact(temp_dir_name, temp_dir)
        payload = copy.deepcopy(meta_props)
        if label_column_names:
            payload["label_column"] = label_column_names[0]

        response = requests.post(
            self._client._href_definitions.get_published_models_href(),
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._process_sw_spec_error(
                response, meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            )

        result = self._handle_response(201, "creating model", response)
        model_id = self._get_required_element_from_dict(
            result, "model_details", ["metadata", "id"]
        )

        with open(path_to_archive, "rb") as f:
            qparams = self._client._params()

            qparams["content_format"] = "native"
            qparams["version"] = self._client.version_param
            # update the content path for the Auto-ai model.

            response = requests.put(
                self._client._href_definitions.get_published_model_content_href(
                    model_id
                ),
                data=f,
                params=qparams,
                headers=self._client._get_headers(
                    content_type="application/octet-stream"
                ),
            )

            if response.status_code != 200 and response.status_code != 201:
                self.delete(model_id)

            self._handle_response(201, "uploading model content", response, False)

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                os.remove(path_to_archive)

            return self.get_details(model_id)

    async def _astore_tf_model(
        self,
        model: Any,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        tf_meta = None
        options = None
        signature = None
        save_format = None
        include_optimizer = None

        if (
            "tf_model_params" in meta_props
            and meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS] is not None
        ):
            tf_meta = copy.deepcopy(
                meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS]
            )
            save_format = tf_meta.get("save_format")
            options = tf_meta.get("options")
            signature = tf_meta.get("signature")
            include_optimizer = tf_meta.get("include_optimizer")

        temp_dir_name, temp_dir = self._save_tensorflow_model_to_directory(
            model, save_format, signature, options, include_optimizer
        )

        path_to_archive = self._model_content_compress_artifact(temp_dir_name, temp_dir)
        payload = copy.deepcopy(meta_props)
        if label_column_names:
            payload["label_column"] = label_column_names[0]

        response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_published_models_href(),
            json=payload,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            await self._aprocess_sw_spec_error(
                response, meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            )

        result = self._handle_response(201, "creating model", response)
        model_id = self._get_required_element_from_dict(
            result, "model_details", ["metadata", "id"]
        )

        qparams = self._client._params()

        qparams["content_format"] = "native"
        qparams["version"] = self._client.version_param
        # update the content path for the Auto-ai model.

        response = await self._client.async_httpx_client.put(
            self._client._href_definitions.get_published_model_content_href(model_id),
            content=AsyncFileReader(path_to_archive),
            params=qparams,
            headers=await self._client._aget_headers(
                content_type="application/octet-stream"
            ),
        )

        if response.status_code != 200 and response.status_code != 201:
            await self.adelete(model_id)

        self._handle_response(201, "uploading model content", response, False)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            os.remove(path_to_archive)

        return await self.aget_details(model_id)

    def _publish_from_object_cloud(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        if (
            self.ConfigurationMetaNames.RUNTIME_ID in meta_props
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. RUNTIME_ID is no longer supported, instead of that "
                "provide SOFTWARE_SPEC_ID in meta_props."
            )
        elif (
            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
            and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in meta_props."
            )

        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                if pipeline is None or training_data is None:
                    raise WMLClientError(
                        "pipeline and training_data are expected for spark models."
                    )
                name = meta_props[self.ConfigurationMetaNames.NAME]
                version = "1.0"
                platform = {"name": "python", "versions": ["3.6"]}
                library_tar = self._save_library_archive(pipeline)
                model_definition_props = {
                    self._client.model_definitions.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                    self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                }
                model_definition_details = self._client.model_definitions.store(
                    library_tar, model_definition_props
                )
                model_definition_id = self._client.model_definitions.get_id(
                    model_definition_details
                )
                # create a pipeline for model definition
                pipeline_metadata = {
                    self._client.pipelines.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                        "doc_type": "pipeline",
                        "version": "2.0",
                        "primary_pipeline": "dlaas_only",
                        "pipelines": [
                            {
                                "id": "dlaas_only",
                                "runtime_ref": "spark",
                                "nodes": [
                                    {
                                        "id": "repository",
                                        "type": "model_node",
                                        "inputs": [],
                                        "outputs": [],
                                        "parameters": {
                                            "model_definition": {
                                                "id": model_definition_id
                                            }
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }

                pipeline_save = self._client.pipelines.store(pipeline_metadata)
                pipeline_id = self._client.pipelines.get_id(pipeline_save)
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "id": pipeline_id
                }
            elif self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
                )
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "id": meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
                }

            if (
                self.ConfigurationMetaNames.SPACE_ID in meta_props
                and meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                is not None
            ):
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
                )
                meta_props["space_id"] = meta_props[
                    self._client.repository.ModelMetaNames.SPACE_ID
                ]
                meta_props.pop(self.ConfigurationMetaNames.SPACE_ID)
            elif self._client.default_project_id is not None:
                meta_props["project_id"] = self._client.default_project_id
            else:
                meta_props["space_id"] = self._client.default_space_id

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
                )
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                }

            if self.ConfigurationMetaNames.MODEL_DEFINITION_ID in meta_props:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.MODEL_DEFINITION_ID,
                    str,
                    False,
                )
                meta_props[
                    self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                ] = {
                    "id": meta_props[
                        self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                    ]
                }

            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith(
                "tensorflow_"
            ):
                return self._store_tf_model(
                    model,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                meta_data = MetaProps(meta_props)
                model_artifact = MLRepositoryArtifact(
                    model,
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    training_data=training_data,
                    training_target=training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param=query_param_for_repo_client
                )

                return self.get_details(saved_model.uid)  # type: ignore[attr-defined]

        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)

    async def _apublish_from_object_cloud(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud asynchronously."""
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        if (
            self.ConfigurationMetaNames.RUNTIME_ID in meta_props
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. RUNTIME_ID is no longer supported, instead of that "
                "provide SOFTWARE_SPEC_ID in meta_props."
            )
        elif (
            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
            and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in meta_props."
            )

        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                if pipeline is None or training_data is None:
                    raise WMLClientError(
                        "pipeline and training_data are expected for spark models."
                    )
                name = meta_props[self.ConfigurationMetaNames.NAME]
                version = "1.0"
                platform = {"name": "python", "versions": ["3.6"]}
                library_tar = self._save_library_archive(pipeline)
                model_definition_props = {
                    self._client.model_definitions.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                    self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                }
                model_definition_details = await self._client.model_definitions.astore(
                    library_tar, model_definition_props
                )
                model_definition_id = self._client.model_definitions.get_id(
                    model_definition_details
                )
                # create a pipeline for model definition
                pipeline_metadata = {
                    self._client.pipelines.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                        "doc_type": "pipeline",
                        "version": "2.0",
                        "primary_pipeline": "dlaas_only",
                        "pipelines": [
                            {
                                "id": "dlaas_only",
                                "runtime_ref": "spark",
                                "nodes": [
                                    {
                                        "id": "repository",
                                        "type": "model_node",
                                        "inputs": [],
                                        "outputs": [],
                                        "parameters": {
                                            "model_definition": {
                                                "id": model_definition_id
                                            }
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }

                pipeline_save = await self._client.pipelines.astore(pipeline_metadata)
                pipeline_id = self._client.pipelines.get_id(pipeline_save)
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "id": pipeline_id
                }
            elif self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
                )
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "id": meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
                }

            if (
                self.ConfigurationMetaNames.SPACE_ID in meta_props
                and meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                is not None
            ):
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
                )
                meta_props["space_id"] = meta_props[
                    self._client.repository.ModelMetaNames.SPACE_ID
                ]
                meta_props.pop(self.ConfigurationMetaNames.SPACE_ID)
            elif self._client.default_project_id is not None:
                meta_props["project_id"] = self._client.default_project_id
            else:
                meta_props["space_id"] = self._client.default_space_id

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
                )
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                }

            if self.ConfigurationMetaNames.MODEL_DEFINITION_ID in meta_props:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.MODEL_DEFINITION_ID,
                    str,
                    False,
                )
                meta_props[
                    self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                ] = {
                    "id": meta_props[
                        self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                    ]
                }

            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith(
                "tensorflow_"
            ):
                return await self._astore_tf_model(
                    model,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                meta_data = MetaProps(meta_props)
                model_artifact = MLRepositoryArtifact(
                    model,
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    training_data=training_data,
                    training_target=training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param=query_param_for_repo_client
                )

                return await self.aget_details(saved_model.uid)  # type: ignore[attr-defined]

        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)

    def _publish_from_file(
        self,
        model: str,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        version: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store saved model into Watson Machine Learning repository on IBM Cloud."""
        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props:
            raise WMLClientError(
                "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in metaprop."
            )

        if version:
            # check if artifactid is passed
            Models._validate_type(artifactid, "artifactid", str, True)
            return self._publish_from_archive(
                model,
                meta_props,
                version=version,
                artifactid=artifactid,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        model_filepath = model
        if os.path.isdir(model):
            # TODO this part is ugly, but will work. In final solution this will be removed
            if "tensorflow" in meta_props[self.ConfigurationMetaNames.TYPE]:
                return self._publish_from_tensorflow_dir(
                    model, meta_props, feature_names, label_column_names
                )

            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )
            if "caffe" in meta_props[self.ConfigurationMetaNames.TYPE]:
                raise WMLClientError(
                    f"Invalid model file path specified for: '{meta_props[self.ConfigurationMetaNames.TYPE]}'."
                )

            loaded_model = load_model_from_directory(
                meta_props[self.ConfigurationMetaNames.TYPE], model
            )

            if self._client.CLOUD_PLATFORM_SPACES:
                saved_model = self._publish_from_object_cloud(
                    loaded_model,
                    meta_props,
                    training_data,
                    training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                saved_model = self._publish_from_object(
                    loaded_model,
                    meta_props,
                    training_data,
                    training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )

            return saved_model
        elif model_filepath.endswith(".pmml"):
            raise WMLClientError(
                "The file name has an unsupported extension. Rename the file with a .xml extension."
            )
        elif model_filepath.endswith(".xml"):
            try:
                # New V4 cloud flow
                input_meta_data = copy.deepcopy(
                    self._create_cloud_model_payload(
                        meta_props,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                )
                meta_data = MetaProps(input_meta_data)

                model_artifact = MLRepositoryArtifact(
                    str(model_filepath),
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )

                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param_for_repo_client
                )
            except Exception as e:
                raise WMLClientError("Publishing model failed.", e)
            else:
                return self.get_details(saved_model.uid)  # type: ignore[attr-defined]
        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            return self._publish_from_archive(
                model,
                meta_props,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )
        elif (
            model_filepath.endswith(".json")
            and self.ConfigurationMetaNames.TYPE in meta_props
            and meta_props[self.ConfigurationMetaNames.TYPE]
            in {"xgboost_1.3", "xgboost_1.5"}
        ):
            # validation
            with open(model, "rb") as file:
                try:
                    json.loads(file.read())
                except Exception:
                    raise WMLClientError(
                        "Json file has invalid content. Please validate if it was generated with xgboost>=1.3."
                    )

            output_filename = model.replace(".json", ".tar.gz")

            try:
                with tarfile.open(output_filename, "w:gz") as tar:
                    tar.add(model, arcname=os.path.basename(model))

                return self._publish_from_archive(
                    output_filename,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            finally:
                os.remove(output_filename)
        else:
            raise WMLClientError(
                f"Saving trained model in repository failed. '{model_filepath}' file does not have valid format"
            )

    async def _apublish_from_file(
        self,
        model: str,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        version: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store saved model into Watson Machine Learning repository on IBM Cloud."""
        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props:
            raise WMLClientError(
                "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in metaprop."
            )

        if version:
            # check if artifactid is passed
            Models._validate_type(artifactid, "artifactid", str, True)
            return await self._apublish_from_archive(
                model,
                meta_props,
                version=version,
                artifactid=artifactid,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        model_filepath = model
        if os.path.isdir(model):
            # TODO this part is ugly, but will work. In final solution this will be removed
            if "tensorflow" in meta_props[self.ConfigurationMetaNames.TYPE]:
                return await self._apublish_from_tensorflow_dir(
                    model, meta_props, feature_names, label_column_names
                )

            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )
            if "caffe" in meta_props[self.ConfigurationMetaNames.TYPE]:
                raise WMLClientError(
                    f"Invalid model file path specified for: '{meta_props[self.ConfigurationMetaNames.TYPE]}'."
                )

            loaded_model = load_model_from_directory(
                meta_props[self.ConfigurationMetaNames.TYPE], model
            )

            if self._client.CLOUD_PLATFORM_SPACES:
                saved_model = await self._apublish_from_object_cloud(
                    loaded_model,
                    meta_props,
                    training_data,
                    training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                saved_model = await self._apublish_from_object(
                    loaded_model,
                    meta_props,
                    training_data,
                    training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )

            return saved_model
        elif model_filepath.endswith(".pmml"):
            raise WMLClientError(
                "The file name has an unsupported extension. Rename the file with a .xml extension."
            )
        elif model_filepath.endswith(".xml"):
            try:
                # New V4 cloud flow
                input_meta_data = copy.deepcopy(
                    self._create_cloud_model_payload(
                        meta_props,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                )
                meta_data = MetaProps(input_meta_data)

                model_artifact = MLRepositoryArtifact(
                    str(model_filepath),
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )

                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param_for_repo_client
                )
            except Exception as e:
                raise WMLClientError("Publishing model failed.", e)
            else:
                return await self.aget_details(saved_model.uid)  # type: ignore[attr-defined]
        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            return await self._apublish_from_archive(
                model,
                meta_props,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )
        elif (
            model_filepath.endswith(".json")
            and self.ConfigurationMetaNames.TYPE in meta_props
            and meta_props[self.ConfigurationMetaNames.TYPE]
            in {"xgboost_1.3", "xgboost_1.5"}
        ):
            # validation
            with open(model, "rb") as file:
                try:
                    json.loads(file.read())
                except Exception:
                    raise WMLClientError(
                        "Json file has invalid content. Please validate if it was generated with xgboost>=1.3."
                    )

            output_filename = model.replace(".json", ".tar.gz")

            try:
                with tarfile.open(output_filename, "w:gz") as tar:
                    tar.add(model, arcname=os.path.basename(model))

                return await self._apublish_from_archive(
                    output_filename,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            finally:
                os.remove(output_filename)
        else:
            raise WMLClientError(
                f"Saving trained model in repository failed. '{model_filepath}' file does not have valid format"
            )

    @staticmethod
    def _raise_warning_for_software_spec_error(sw_spec_details: dict) -> None:
        sw_spec = get_from_json(sw_spec_details, ["metadata", "name"])
        spec_lifecycle = get_from_json(sw_spec_details, ["metadata", "life_cycle"], {})

        if replacement := spec_lifecycle.get("replacement_name"):
            replacement_str = (
                f" Use replacement software specification instead: {replacement}"
            )
        else:
            replacement_str = ""

        if spec_lifecycle.get("retired"):
            if retired_since := spec_lifecycle.get("retired_since_version"):
                retired_software_spec_warning = f"Software specification `{sw_spec}` is retired since version {retired_since}."
            else:
                retired_software_spec_warning = (
                    f"Software specification `{sw_spec}` is retired."
                )

            warn(
                retired_software_spec_warning + replacement_str,
                PendingDeprecationWarning,
            )
        elif spec_lifecycle.get("deprecated"):
            if deprecated_since := spec_lifecycle.get("deprecated_since_version"):
                deprecated_software_spec_warning = f"Software specification `{sw_spec}` is deprecated since version {deprecated_since}."
            else:
                deprecated_software_spec_warning = (
                    f"Software specification `{sw_spec}` is deprecated."
                )

            warn(
                deprecated_software_spec_warning + replacement_str,
                DeprecationWarning,
            )

    def _process_sw_spec_error(self, response, sw_spec_id: str) -> None:
        if response.status_code != 400:
            return

        if all(
            message not in response.text
            for message in [
                "Invalid request entity: Unsupported software specification",
                "Unsupported model type and software specification combination",
            ]
        ):
            return

        sw_spec_details = self._client.software_specifications.get_details(sw_spec_id)

        self._raise_warning_for_software_spec_error(sw_spec_details)

    async def _aprocess_sw_spec_error(self, response, sw_spec_id: str) -> None:
        if response.status_code != 400:
            return

        if all(
            message not in response.text
            for message in [
                "Invalid request entity: Unsupported software specification",
                "Unsupported model type and software specification combination",
            ]
        ):
            return

        sw_spec_details = await self._client.software_specifications.aget_details(
            sw_spec_id
        )

        self._raise_warning_for_software_spec_error(sw_spec_details)

    def _publish_empty_model_asset(
        self,
        meta_props: dict[str, Any],
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """
        The method creates model asset without uploading model content.
        """

        model_payload = self._create_cloud_model_payload(
            meta_props,
            label_column_names=label_column_names,
        )

        creation_response = requests.post(
            self._client._href_definitions.get_published_models_href(),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=model_payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._process_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201, "creating new model", creation_response
        )

        return model_details

    async def _apublish_empty_model_asset(
        self,
        meta_props: dict[str, Any],
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """
        The method creates model asset without uploading model content asynchronously.
        """

        model_payload = self._create_cloud_model_payload(
            meta_props,
            label_column_names=label_column_names,
        )

        creation_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_published_models_href(),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=model_payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            await self._aprocess_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201, "creating new model", creation_response
        )

        return model_details

    def _store_prompt_tuning_model(self, training_id: str, meta_props: dict) -> dict:
        # import here to avoid circular import
        from ibm_watsonx_ai.foundation_models.utils.utils import (
            load_request_json,  # pylint: disable=import-outside-toplevel
        )

        model_request_json = load_request_json(
            run_id=training_id, api_client=self._client
        )
        if meta_props:
            model_request_json.update(meta_props)

        creation_response = requests.post(
            self._client._href_definitions.get_published_models_href(),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=model_request_json,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._process_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            202, "creating new model", creation_response
        )

        return self._wait_for_content_import_completion(model_details)

    async def _astore_prompt_tuning_model(
        self, training_id: str, meta_props: dict
    ) -> dict:
        # import here to avoid circular import
        from ibm_watsonx_ai.foundation_models.utils.utils import (
            aload_request_json,  # pylint: disable=import-outside-toplevel
        )

        model_request_json = await aload_request_json(
            run_id=training_id, api_client=self._client
        )
        if meta_props:
            model_request_json.update(meta_props)

        creation_response = await self._client.async_httpx_client.post(
            self._client._href_definitions.get_published_models_href(),
            headers=await self._client._aget_headers(),
            params=self._client._params(),
            json=model_request_json,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            await self._aprocess_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            202, "creating new model", creation_response
        )

        return await self._await_for_content_import_completion(model_details)

    def _create_custom_model_payload(
        self, model: str, meta_props: dict[str, Any]
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)

        Models._validate_type(model, "model", str, True)

        if self.ConfigurationMetaNames.TYPE in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
            )
            metadata[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            }

        metadata[self.ConfigurationMetaNames.FOUNDATION_MODEL] = {"model_id": model}

        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            metadata["project_id"] = self._client.default_project_id
        else:
            raise WMLClientError(
                Messages.get_message(
                    message_id="it_is_mandatory_to_set_the_space_project_id"
                )
            )

        if self._client.CLOUD_PLATFORM_SPACES:
            if self.ConfigurationMetaNames.MODEL_LOCATION not in meta_props:
                raise WMLClientError("model_location missing in meta_props")

            conn_id = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "connection_id"
            )
            bucket = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "bucket"
            )
            path = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "file_path"
            )
            conn_type = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "type"
            )

            if conn_id is None:
                raise WMLClientError("connection_id missing in meta_props")

            if bucket is None:
                raise WMLClientError("bucket missing in meta_props")

            if path is None:
                raise WMLClientError("file_path missing in meta_props")

            model_location = {
                "type": conn_type if conn_type else "connection_asset",
                "connection": {"id": conn_id},
                "location": {"bucket": bucket, "file_path": path},
            }

            metadata[self.ConfigurationMetaNames.FOUNDATION_MODEL][
                self.ConfigurationMetaNames.MODEL_LOCATION
            ] = model_location

            if self.ConfigurationMetaNames.MODEL_LOCATION in metadata:
                del metadata[self.ConfigurationMetaNames.MODEL_LOCATION]

        return metadata

    def _store_custom_foundation_model(
        self, model: str, meta_props: dict[str, Any]
    ) -> dict[str, Any]:
        payload = self._create_custom_model_payload(model=model, meta_props=meta_props)

        creation_response = requests.post(
            url=self._client._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
            json=payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._process_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return self._wait_for_content_import_completion(model_details)

    async def _astore_custom_foundation_model(
        self, model: str, meta_props: dict[str, Any]
    ) -> dict[str, Any]:
        payload = self._create_custom_model_payload(model=model, meta_props=meta_props)

        creation_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=await self._client._aget_headers(),
            json=payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            await self._aprocess_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return await self._await_for_content_import_completion(model_details)

    def _create_type_of_model_payload(
        self,
        model: str,
        meta_props: dict[str, Any],
        model_type: Literal["curated", "base"],
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)

        Models._validate_type(model, "model", str, True)

        if self.ConfigurationMetaNames.TYPE in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )

        if model_type == "curated" and not model.endswith("-curated"):
            model += "-curated"
        elif (
            model_type == "base"
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
            )
            metadata[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
            }

        metadata[self.ConfigurationMetaNames.FOUNDATION_MODEL] = {"model_id": model}

        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            metadata["project_id"] = self._client.default_project_id
        else:
            raise WMLClientError(
                Messages.get_message(
                    message_id="it_is_mandatory_to_set_the_space_project_id"
                )
            )

        return metadata

    def _store_type_of_foundation_model(
        self,
        model: str,
        meta_props: dict[str, Any],
        model_type: Literal["curated", "base"],
    ) -> dict[str, Any]:
        payload = self._create_type_of_model_payload(
            model=model, meta_props=meta_props, model_type=model_type
        )

        creation_response = requests.post(
            url=self._client._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
            json=payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._process_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return self._wait_for_content_import_completion(model_details)

    async def _astore_type_of_foundation_model(
        self,
        model: str,
        meta_props: dict[str, Any],
        model_type: Literal["curated", "base"],
    ) -> dict[str, Any]:
        payload = self._create_type_of_model_payload(
            model=model, meta_props=meta_props, model_type=model_type
        )

        creation_response = await self._client.async_httpx_client.post(
            url=self._client._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=await self._client._aget_headers(),
            json=payload,
        )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            await self._aprocess_sw_spec_error(
                creation_response,
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID],
            )

        model_details = self._handle_response(
            201 if creation_response.status_code == 201 else 202,
            "creating new model",
            creation_response,
        )

        return await self._await_for_content_import_completion(model_details)

    def _get_last_run_metrics_name(self, training_id: str) -> str:
        run_metrics = self._client.training.get_metrics(training_id=training_id)
        return get_from_json(run_metrics[-1], ["context", "intermediate_model", "name"])

    async def _aget_last_run_metrics_name(self, training_id: str) -> str:
        run_metrics = await self._client.training.aget_metrics(training_id=training_id)
        return get_from_json(run_metrics[-1], ["context", "intermediate_model", "name"])

    def _store_from_object(
        self,
        model: MLModelType,
        meta_props: dict[str, Any] | None,
        training_data: TrainingDataType | None,
        training_target: TrainingTargetType | None,
        pipeline: PipelineType | None,
        version: bool,
        feature_names: numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None,
        label_column_names: LabelColumnNamesType | None,
        experiment_metadata: dict[str, Any] | None,
        training_id: str | None,
    ) -> dict:
        if version is True:
            raise WMLClientError(
                "Unsupported type: object for param model. Supported types: path to saved model, training ID"
            )

        if not experiment_metadata and not training_id:
            return self._publish_from_object_cloud(
                model=model,
                meta_props=meta_props,
                training_data=training_data,
                training_target=training_target,
                pipeline=pipeline,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        if experiment_metadata:
            training_id = get_autoai_run_id_from_experiment_metadata(
                experiment_metadata
            )

        # Note: validate if training_id is from AutoAI experiment
        run_params = self._client.training.get_details(
            training_id=training_id, _internal=True
        )
        pipeline_id = get_from_json(run_params, ["entity", "pipeline", "id"])

        pipeline_details, pipeline_nodes_list = {}, []
        if pipeline_id:
            pipeline_details = self._client.pipelines.get_details(pipeline_id)
            pipeline_nodes_list = get_from_json(
                pipeline_details, ["entity", "document", "pipelines"], []
            )

        if len(pipeline_nodes_list) == 0 or pipeline_nodes_list[0]["id"] != "autoai":
            raise WMLClientError(
                "Parameter training_id or experiment_metadata is not connected to AutoAI training"
            )

        if is_lale_pipeline(model):
            model = model.export_to_sklearn_pipeline()

        with catch_warnings():
            simplefilter("ignore", category=DeprecationWarning)
            schema, artifact_name = prepare_auto_ai_model_to_publish(
                pipeline_model=model,
                run_params=run_params,
                run_id=training_id,
                api_client=self._client,
            )

        new_meta_props = {
            self._client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
            self._client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: self._client.software_specifications.get_id_by_name(
                "hybrid_0.1"
            ),
        }

        results_reference = DataConnection._from_dict(
            run_params["entity"]["results_reference"]
        )
        results_reference.set_client(self._client)

        request_json = download_request_json(
            run_params=run_params,
            model_name=self._get_last_run_metrics_name(
                training_id=cast(str, training_id)
            ),
            api_client=self._client,
            results_reference=results_reference,
        )
        if request_json:
            if "schemas" in request_json:
                new_meta_props["schemas"] = request_json["schemas"]
            if "hybrid_pipeline_software_specs" in request_json:
                new_meta_props["hybrid_pipeline_software_specs"] = request_json[
                    "hybrid_pipeline_software_specs"
                ]

        if experiment_metadata:
            if "prediction_column" in experiment_metadata:
                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] = (
                    experiment_metadata.get("prediction_column")
                )

            if "training_data_references" in experiment_metadata:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = [
                    e._to_dict() if isinstance(e, DataConnection) else e
                    for e in experiment_metadata.get("training_data_references", [])
                ]

                if new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ]:
                    new_meta_props[
                        self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                    ][0]["schema"] = schema

            if "test_data_references" in experiment_metadata:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TEST_DATA_REFERENCES
                ] = [
                    e._to_dict() if isinstance(e, DataConnection) else e
                    for e in experiment_metadata.get("test_data_references", [])
                ]
        elif training_id:
            label_column = None
            for node in pipeline_details["entity"]["document"]["pipelines"][0]["nodes"]:
                if "automl" in node["id"] or "autoai" in node["id"]:
                    label_column = get_from_json(
                        node, ["parameters", "optimization", "label"]
                    )

            if label_column is not None:
                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] = (
                    label_column
                )

            # TODO Is training_data_references and test_data_references needed in meta props??
            if "training_data_references" in run_params["entity"]:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = run_params["entity"]["training_data_references"]

            if "test_data_references" in run_params["entity"]:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = run_params["entity"]["test_data_references"]

        if pipeline_id := get_from_json(run_params, ["entity", "pipeline", "id"]):
            new_meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = (
                pipeline_id
            )

        new_meta_props.update(meta_props)

        return self.store(model=artifact_name, meta_props=new_meta_props)

    async def _astore_from_object(
        self,
        model: MLModelType,
        meta_props: dict[str, Any] | None,
        training_data: TrainingDataType | None,
        training_target: TrainingTargetType | None,
        pipeline: PipelineType | None,
        version: bool,
        feature_names: numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None,
        label_column_names: LabelColumnNamesType | None,
        experiment_metadata: dict[str, Any] | None,
        training_id: str | None,
    ) -> dict:
        if version is True:
            raise WMLClientError(
                "Unsupported type: object for param model. Supported types: path to saved model, training ID"
            )

        if not experiment_metadata and not training_id:
            return await self._apublish_from_object_cloud(
                model=model,
                meta_props=meta_props,
                training_data=training_data,
                training_target=training_target,
                pipeline=pipeline,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        if experiment_metadata:
            training_id = get_autoai_run_id_from_experiment_metadata(
                experiment_metadata
            )

        # Note: validate if training_id is from AutoAI experiment
        run_params = await self._client.training.aget_details(
            training_id=training_id, _internal=True
        )
        pipeline_id = get_from_json(run_params, ["entity", "pipeline", "id"])

        pipeline_details, pipeline_nodes_list = {}, []
        if pipeline_id:
            pipeline_details = await self._client.pipelines.aget_details(pipeline_id)
            pipeline_nodes_list = get_from_json(
                pipeline_details, ["entity", "document", "pipelines"], []
            )

        if len(pipeline_nodes_list) == 0 or pipeline_nodes_list[0]["id"] != "autoai":
            raise WMLClientError(
                "Parameter training_id or experiment_metadata is not connected to AutoAI training"
            )

        if is_lale_pipeline(model):
            model = model.export_to_sklearn_pipeline()

        with catch_warnings():
            simplefilter("ignore", category=DeprecationWarning)
            schema, artifact_name = await aprepare_auto_ai_model_to_publish(
                pipeline_model=model,
                run_params=run_params,
                run_id=training_id,
                api_client=self._client,
            )

        new_meta_props = {
            self._client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
            self._client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: await self._client.software_specifications.aget_id_by_name(
                "hybrid_0.1"
            ),
        }

        results_reference = DataConnection._from_dict(
            run_params["entity"]["results_reference"]
        )
        results_reference.set_client(self._client)

        request_json = await adownload_request_json(
            run_params=run_params,
            model_name=await self._aget_last_run_metrics_name(
                training_id=cast(str, training_id)
            ),
            api_client=self._client,
            results_reference=results_reference,
        )
        if request_json:
            if "schemas" in request_json:
                new_meta_props["schemas"] = request_json["schemas"]
            if "hybrid_pipeline_software_specs" in request_json:
                new_meta_props["hybrid_pipeline_software_specs"] = request_json[
                    "hybrid_pipeline_software_specs"
                ]

        if experiment_metadata:
            if "prediction_column" in experiment_metadata:
                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] = (
                    experiment_metadata.get("prediction_column")
                )

            if "training_data_references" in experiment_metadata:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = [
                    e._to_dict() if isinstance(e, DataConnection) else e
                    for e in experiment_metadata.get("training_data_references", [])
                ]

                if new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ]:
                    new_meta_props[
                        self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                    ][0]["schema"] = schema

            if "test_data_references" in experiment_metadata:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TEST_DATA_REFERENCES
                ] = [
                    e._to_dict() if isinstance(e, DataConnection) else e
                    for e in experiment_metadata.get("test_data_references", [])
                ]
        elif training_id:
            label_column = None
            for node in pipeline_details["entity"]["document"]["pipelines"][0]["nodes"]:
                if "automl" in node["id"] or "autoai" in node["id"]:
                    label_column = get_from_json(
                        node, ["parameters", "optimization", "label"]
                    )

            if label_column is not None:
                new_meta_props[self._client.repository.ModelMetaNames.LABEL_FIELD] = (
                    label_column
                )

            # TODO Is training_data_references and test_data_references needed in meta props??
            if "training_data_references" in run_params["entity"]:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = run_params["entity"]["training_data_references"]

            if "test_data_references" in run_params["entity"]:
                new_meta_props[
                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                ] = run_params["entity"]["test_data_references"]

        if pipeline_id := get_from_json(run_params, ["entity", "pipeline", "id"]):
            new_meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = (
                pipeline_id
            )

        new_meta_props.update(meta_props)

        return await self.astore(model=artifact_name, meta_props=new_meta_props)

    def _store_from_str(
        self,
        model: str,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None,
        training_target: TrainingTargetType | None,
        version: bool,
        artifactid: str | None,
        feature_names: numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None,
        label_column_names: LabelColumnNamesType | None,
        subtrainingId: str | None,
        round_number: int | None,
        experiment_metadata: dict[str, Any] | None,
        training_id: str | None,
    ) -> dict:
        if os.path.sep in model and (
            model.endswith(".pickle") or model.endswith("pipeline-model.json")
        ):
            # AUTO AI Trained model
            # pipeline-model.json is needed for OBM + KB
            return self._store_auto_ai_model(
                model_path=model,
                meta_props=meta_props,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        if model.startswith("Pipeline_") and (experiment_metadata or training_id):
            if experiment_metadata:
                training_id = get_autoai_run_id_from_experiment_metadata(
                    experiment_metadata
                )

            # Note: validate if training_id is from AutoAI experiment
            run_params = self._client.training.get_details(
                training_id=training_id, _internal=True
            )

            # raise an error when TS pipeline is discarded one
            check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

            # Note: We need to fetch credentials when 'container' is the type
            if run_params["entity"]["results_reference"]["type"] == "container":
                data_connection = DataConnection._from_dict(
                    _dict=run_params["entity"]["results_reference"]
                )
                data_connection.set_client(self._client)
            else:
                data_connection = None
            # --- end note

            artifact_name, model_props = (
                prepare_auto_ai_model_to_publish_normal_scenario(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=training_id,
                    api_client=self._client,
                    result_reference=data_connection,
                )
            )
            model_props.update(meta_props)

            return self.store(artifact_name, model_props)

        if os.path.sep in model or os.path.isfile(model) or os.path.isdir(model):
            if not os.path.isfile(model) and not os.path.isdir(model):
                raise WMLClientError(
                    "Invalid path: neither file nor directory exists under this path: '{}'.".format(
                        model
                    )
                )

            return self._publish_from_file(
                model=model,
                meta_props=meta_props,
                training_data=training_data,
                training_target=training_target,
                version=version,
                artifactid=artifactid,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        return self._publish_from_training(
            model_id=model,
            meta_props=meta_props,
            subtrainingId=subtrainingId,
            feature_names=feature_names,
            label_column_names=label_column_names,
            round_number=round_number,
        )

    async def _astore_from_str(
        self,
        model: str,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None,
        training_target: TrainingTargetType | None,
        version: bool,
        artifactid: str | None,
        feature_names: numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None,
        label_column_names: LabelColumnNamesType | None,
        subtrainingId: str | None,
        round_number: int | None,
        experiment_metadata: dict[str, Any] | None,
        training_id: str | None,
    ) -> dict:
        if os.path.sep in model and (
            model.endswith(".pickle") or model.endswith("pipeline-model.json")
        ):
            # AUTO AI Trained model
            # pipeline-model.json is needed for OBM + KB
            return await self._astore_auto_ai_model(
                model_path=model,
                meta_props=meta_props,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        if model.startswith("Pipeline_") and (experiment_metadata or training_id):
            if experiment_metadata:
                training_id = get_autoai_run_id_from_experiment_metadata(
                    experiment_metadata
                )

            # Note: validate if training_id is from AutoAI experiment
            run_params = await self._client.training.aget_details(
                training_id=training_id, _internal=True
            )

            # raise an error when TS pipeline is discarded one
            check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

            # Note: We need to fetch credentials when 'container' is the type
            if run_params["entity"]["results_reference"]["type"] == "container":
                data_connection = DataConnection._from_dict(
                    _dict=run_params["entity"]["results_reference"]
                )
                data_connection.set_client(self._client)
            else:
                data_connection = None
            # --- end note

            (
                artifact_name,
                model_props,
            ) = await aprepare_auto_ai_model_to_publish_normal_scenario(
                pipeline_model=model,
                run_params=run_params,
                run_id=training_id,
                api_client=self._client,
                result_reference=data_connection,
            )
            model_props.update(meta_props)

            return await self.astore(artifact_name, model_props)

        if os.path.sep in model or os.path.isfile(model) or os.path.isdir(model):
            if not os.path.isfile(model) and not os.path.isdir(model):
                raise WMLClientError(
                    "Invalid path: neither file nor directory exists under this path: '{}'.".format(
                        model
                    )
                )

            return await self._apublish_from_file(
                model=model,
                meta_props=meta_props,
                training_data=training_data,
                training_target=training_target,
                version=version,
                artifactid=artifactid,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        return await self._apublish_from_training(
            model_id=model,
            meta_props=meta_props,
            subtrainingId=subtrainingId,
            feature_names=feature_names,
            label_column_names=label_column_names,
            round_number=round_number,
        )

    def _get_store_model_type(
        self, meta_props: dict | None
    ) -> Literal["custom_model", "curated_model", "base_foundation_model"] | None:
        if not isinstance(meta_props, dict):
            return None

        meta_props_type = meta_props.get(self.ConfigurationMetaNames.TYPE, "")

        if meta_props_type.startswith("custom_foundation_model"):
            if self._client.CPD_version < 4.8:
                raise WMLClientError(
                    Messages.get_message(">= 4.8", message_id="invalid_cpd_version")
                )

            return "custom_model"

        if meta_props_type.startswith("curated_foundation_model"):
            if not self._client.CLOUD_PLATFORM_SPACES:
                raise WMLClientError(
                    error_msg="Deploy on Demand is unsupported for this release."
                )

            return "curated_model"

        if meta_props_type.startswith("base_foundation_model"):
            if self._client.CPD_version < 5.0:
                raise WMLClientError(
                    Messages.get_message(">= 5.0", message_id="invalid_cpd_version")
                )

            return "base_foundation_model"

        return None

    def _prepare_store_meta_props(self, meta_props: dict | None) -> dict:
        if meta_props is None:
            meta_props = {}

        meta_props = copy.deepcopy(meta_props)

        if training_data_references := meta_props.get(
            self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
        ):
            converted_data_references = []

            self._validate_type(
                training_data_references,
                "training_data_references",
                expected_type=list,
                mandatory=False,
            )

            for data_reference in training_data_references:
                data_reference_dict = (
                    data_reference.to_dict()
                    if isinstance(data_reference, DataConnection)
                    else data_reference
                )

                self._validate_type(
                    data_reference_dict.get("type"),
                    "training_data_references.type",
                    expected_type=str,
                    mandatory=True,
                )

                converted_data_references.append(data_reference_dict)

            meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES] = (
                converted_data_references
            )

        if (
            self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES in meta_props
            and self.ConfigurationMetaNames.INPUT_DATA_SCHEMA not in meta_props
            and meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0].get(
                "schema"
            )
        ):
            try:
                training_schema = meta_props[
                    self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
                ][0]["schema"]

                if not meta_props.get(
                    self.ConfigurationMetaNames.LABEL_FIELD
                ) and training_schema.get("fields"):
                    fields = training_schema["fields"]
                    target_fields = [
                        field["name"]
                        for field in fields
                        if field["metadata"].get("modeling_role") == "target"
                    ]
                    if target_fields:
                        meta_props[self.ConfigurationMetaNames.LABEL_FIELD] = (
                            target_fields[0]
                        )

                if meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD):
                    input_data_schema = {
                        "fields": [
                            field
                            for field in training_schema["fields"]
                            if self.ConfigurationMetaNames.LABEL_FIELD not in meta_props
                            or field["name"]
                            != meta_props[self.ConfigurationMetaNames.LABEL_FIELD]
                        ],
                        "type": "struct",
                        "id": "1",
                    }

                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] = (
                        input_data_schema
                    )
            except Exception:
                pass

        return meta_props

    def store(
        self,
        model: MLModelType = None,
        meta_props: dict[str, Any] | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        version: bool = False,
        artifactid: str | None = None,
        feature_names: (
            numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None
        ) = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict[str, Any] | None = None,
        training_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a model.

        :ref:`Here<save_models>` you can explore how to save external models in correct format.

        :param model: Can be one of following:

            - The train model object:\n
                - scikit-learn
                - xgboost
                - spark (PipelineModel)
            - path to saved model in format:\n
                - tensorflow / keras (.tar.gz)
                - pmml (.xml)
                - scikit-learn (.tar.gz)
                - spss (.str)
                - spark (.tar.gz)
                - xgboost (.tar.gz)
            - directory containing model file(s):\n
                - scikit-learn
                - xgboost
                - tensorflow
            - unique ID of the trained model
            - LLM name
        :type model: str (for filename, path, or LLM name) or object (corresponding to model type)

        :param meta_props: metadata of the models configuration. To see available meta names, use:

            .. code-block:: python

                client._models.ConfigurationMetaNames.get()

        :type meta_props: dict, optional

        :param training_data: Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array
            supported for scikit-learn models
        :type training_data: spark dataframe, pandas dataframe, numpy.ndarray or array, optional

        :param training_target: array with labels required for scikit-learn models
        :type training_target: array, optional

        :param pipeline: pipeline required for spark mllib models
        :type pipeline: object, optional

        :param feature_names: feature names for the training data in case of Scikit-Learn/XGBoost models,
            this is applicable only in the case where the training data is not of type - pandas.DataFrame
        :type feature_names: numpy.ndarray or list, optional

        :param label_column_names: label column names of the trained Scikit-Learn/XGBoost models
        :type label_column_names: numpy.ndarray or list, optional

        :param round_number: round number of a Federated Learning experiment that has been configured to save
            intermediate models, this applies when model is a training id
        :type round_number: int, optional

        :param experiment_metadata: metadata retrieved from the experiment that created the model
        :type experiment_metadata: dict, optional

        :param training_id: Run id of AutoAI or TuneExperiment experiment.
        :type training_id: str, optional

        :return: metadata of the created model
        :rtype: dict

        .. note::

            * For a keras model, model content is expected to contain a .h5 file and an archived version of it.

            * `feature_names` is an optional argument containing the feature names for the training data
              in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list.
              This is applicable only in the case where the training data is not of type - pandas.DataFrame.

            * If the `training_data` is of type pandas.DataFrame and `feature_names` are provided,
              `feature_names` are ignored.

            * For INPUT_DATA_SCHEMA meta prop use list even when passing single input data schema. You can provide
              multiple schemas as dictionaries inside a list.

            * More details about Foundation Models you can find :ref:`here<foundation_models>`.

        **Examples**

        .. code-block:: python

            stored_model_details = client._models.store(model, name)

        In more complicated cases you should create proper metadata, similar to this one:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'scikit-learn_0.23'
            }

        In case when you want to provide input data schema of the model, you can provide it as part of meta:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'spss-modeler_18.1',
                client._models.ConfigurationMetaNames.INPUT_DATA_SCHEMA: [
                    {
                        'id': 'test',
                        'type': 'list',
                        'fields': [
                            {'name': 'age', 'type': 'float'},
                            {'name': 'sex', 'type': 'float'},
                            {'name': 'fbs', 'type': 'float'},
                            {'name': 'restbp', 'type': 'float'}
                        ]
                    },
                    {
                        'id': 'test2',
                        'type': 'list',
                        'fields': [
                            {'name': 'age', 'type': 'float'},
                            {'name': 'sex', 'type': 'float'},
                            {'name': 'fbs', 'type': 'float'},
                            {'name': 'restbp', 'type': 'float'}
                        ]
                    }
                ]
            }

        ``store()`` method used with a local tar.gz file that contains a model:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_tar_gz, meta_props=metadata, training_data=None)

        ``store()`` method used with a local directory that contains model files:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_model_directory, meta_props=metadata, training_data=None)

        ``store()`` method used with the ID of a trained model:

        .. code-block:: python

            stored_model_details = client._models.store(trained_model_id, meta_props=metadata, training_data=None)

        ``store()`` method used with a pipeline that was generated by an AutoAI experiment:

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction model stored from object'
            }
            stored_model_details = client._models.store(pipeline_model, meta_props=metadata, experiment_metadata=experiment_metadata)

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction Pipeline_1 model'
            }
            stored_model_details = client._models.store(model="Pipeline_1", meta_props=metadata, training_id = training_id)

        Example of storing a prompt tuned model:

        .. code-block:: python

            stored_model_details = client._models.store(training_id = prompt_tuning_run_id)

        Example of storing a custom foundation model:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.0')

            metadata = {
                client.repository.ModelMetaNames.NAME: 'custom FM asset',
                client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0
            }
            stored_model_details = client._models.store(model='mistralai/Mistral-7B-Instruct-v0.2', meta_props=metadata)

        """

        # Import here to avoid circular import
        from ibm_watsonx_ai.foundation_models.utils.utils import (  # pylint: disable=import-outside-toplevel
            is_training_prompt_tuning,
        )

        if (
            self._client.default_space_id is None
            and self._client.default_project_id is None
        ):
            raise WMLClientError(
                "It is mandatory is set the space or project. "
                "Use client.set.default_space(<SPACE_ID>) to set the space "
                "or client.set.default_project(<PROJECT_ID>)."
            )

        if isinstance(meta_props, dict) and (
            "project" in meta_props or "space" in meta_props
        ):
            raise WMLClientError(
                "'project' (MetaNames.PROJECT_ID) and 'space' (MetaNames.SPACE_ID) "
                "meta names are deprecated and considered as invalid. "
                "Instead use client.set.default_space(<SPACE_ID>) to set "
                "the space or client.set.default_project(<PROJECT_ID>)."
            )

        model_type = self._get_store_model_type(meta_props)

        is_prompt_tuned_training = is_training_prompt_tuning(
            training_id, api_client=self._client
        )

        Models._validate_type(
            meta_props,
            "meta_props",
            dict,
            mandatory=not is_prompt_tuned_training,
        )

        is_do_model = (
            (meta_props or {})
            .get(self.ConfigurationMetaNames.TYPE, "")
            .startswith("do-")
        )  # For DO model can be None, see #38648

        Models._validate_type(
            model,
            "model",
            object,
            mandatory=not (is_prompt_tuned_training or is_do_model),
        )

        meta_props = self._prepare_store_meta_props(meta_props)

        is_training_microservice = model is None or (
            isinstance(model, str) and "autoai_sdk" in model
        )

        # note: do not validate meta props when we have them from training microservice (always correct)
        if not is_training_microservice:
            if experiment_metadata or training_id:
                # note: if experiment_metadata are not None it means that the model is created from experiment,
                # and all required information are known from the experiment metadata and the origin
                Models._validate_type(meta_props, "meta_props", dict, True)
                Models._validate_type(meta_props["name"], "meta_props.name", str, True)
            else:
                self.ConfigurationMetaNames._validate(meta_props)

        framework_name = meta_props.get("frameworkName", "").lower()
        if version and framework_name in {"mllib", "wml"}:
            raise WMLClientError(
                "Unsupported framework name: '{}' for creating a model version".format(
                    framework_name
                )
            )

        if training_id and is_prompt_tuned_training:
            saved_model = self._store_prompt_tuning_model(training_id, meta_props)
        elif model is None:
            saved_model = self._publish_empty_model_asset(meta_props)
        elif model_type == "custom_model":
            saved_model = self._store_custom_foundation_model(model, meta_props)
        elif model_type == "curated_model":
            saved_model = self._store_type_of_foundation_model(
                model, meta_props, "curated"
            )
        elif model_type == "base_foundation_model":
            saved_model = self._store_type_of_foundation_model(
                model, meta_props, "base"
            )
        elif isinstance(model, str):
            saved_model = self._store_from_str(
                model,
                meta_props,
                training_data,
                training_target,
                version,
                artifactid,
                feature_names,
                label_column_names,
                subtrainingId,
                round_number,
                experiment_metadata,
                training_id,
            )
        else:
            saved_model = self._store_from_object(
                model,
                meta_props,
                training_data,
                training_target,
                pipeline,
                version,
                feature_names,
                label_column_names,
                experiment_metadata,
                training_id,
            )

        if warning_list := get_from_json(saved_model, ["system", "warnings"]):
            message = warning_list[0].get("message", None)
            print("Note: Warnings!! : ", message)

        return saved_model

    async def astore(
        self,
        model: MLModelType = None,
        meta_props: dict[str, Any] | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        version: bool = False,
        artifactid: str | None = None,
        feature_names: (
            numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None
        ) = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict[str, Any] | None = None,
        training_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a model asynchronously.

        :ref:`Here<save_models>` you can explore how to save external models in correct format.

        :param model: Can be one of following:

            - The train model object:\n
                - scikit-learn
                - xgboost
                - spark (PipelineModel)
            - path to saved model in format:\n
                - tensorflow / keras (.tar.gz)
                - pmml (.xml)
                - scikit-learn (.tar.gz)
                - spss (.str)
                - spark (.tar.gz)
                - xgboost (.tar.gz)
            - directory containing model file(s):\n
                - scikit-learn
                - xgboost
                - tensorflow
            - unique ID of the trained model
            - LLM name
        :type model: str (for filename, path, or LLM name) or object (corresponding to model type)

        :param meta_props: metadata of the models configuration. To see available meta names, use:

            .. code-block:: python

                client._models.ConfigurationMetaNames.get()

        :type meta_props: dict, optional

        :param training_data: Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array
            supported for scikit-learn models
        :type training_data: spark dataframe, pandas dataframe, numpy.ndarray or array, optional

        :param training_target: array with labels required for scikit-learn models
        :type training_target: array, optional

        :param pipeline: pipeline required for spark mllib models
        :type pipeline: object, optional

        :param feature_names: feature names for the training data in case of Scikit-Learn/XGBoost models,
            this is applicable only in the case where the training data is not of type - pandas.DataFrame
        :type feature_names: numpy.ndarray or list, optional

        :param label_column_names: label column names of the trained Scikit-Learn/XGBoost models
        :type label_column_names: numpy.ndarray or list, optional

        :param round_number: round number of a Federated Learning experiment that has been configured to save
            intermediate models, this applies when model is a training id
        :type round_number: int, optional

        :param experiment_metadata: metadata retrieved from the experiment that created the model
        :type experiment_metadata: dict, optional

        :param training_id: Run id of AutoAI or TuneExperiment experiment.
        :type training_id: str, optional

        :return: metadata of the created model
        :rtype: dict

        .. note::

            * For a keras model, model content is expected to contain a .h5 file and an archived version of it.

            * `feature_names` is an optional argument containing the feature names for the training data
              in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list.
              This is applicable only in the case where the training data is not of type - pandas.DataFrame.

            * If the `training_data` is of type pandas.DataFrame and `feature_names` are provided,
              `feature_names` are ignored.

            * For INPUT_DATA_SCHEMA meta prop use list even when passing single input data schema. You can provide
              multiple schemas as dictionaries inside a list.

            * More details about Foundation Models you can find :ref:`here<foundation_models>`.

        **Examples**

        .. code-block:: python

            stored_model_details = await client._models.astore(model, name)

        In more complicated cases you should create proper metadata, similar to this one:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'scikit-learn_0.23'
            }

        In case when you want to provide input data schema of the model, you can provide it as part of meta:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'spss-modeler_18.1',
                client._models.ConfigurationMetaNames.INPUT_DATA_SCHEMA: [
                    {
                        'id': 'test',
                        'type': 'list',
                        'fields': [
                            {'name': 'age', 'type': 'float'},
                            {'name': 'sex', 'type': 'float'},
                            {'name': 'fbs', 'type': 'float'},
                            {'name': 'restbp', 'type': 'float'}
                        ]
                    },
                    {
                        'id': 'test2',
                        'type': 'list',
                        'fields': [
                            {'name': 'age', 'type': 'float'},
                            {'name': 'sex', 'type': 'float'},
                            {'name': 'fbs', 'type': 'float'},
                            {'name': 'restbp', 'type': 'float'}
                        ]
                    }
                ]
            }

        ``astore()`` method used with a local tar.gz file that contains a model:

        .. code-block:: python

            stored_model_details = await client._models.astore(path_to_tar_gz, meta_props=metadata, training_data=None)

        ``astore()`` method used with a local directory that contains model files:

        .. code-block:: python

            stored_model_details = await client._models.astore(path_to_model_directory, meta_props=metadata, training_data=None)

        ``astore()`` method used with the ID of a trained model:

        .. code-block:: python

            stored_model_details = await client._models.astore(trained_model_id, meta_props=metadata, training_data=None)

        ``astore()`` method used with a pipeline that was generated by an AutoAI experiment:

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction model stored from object'
            }
            stored_model_details = await client._models.astore(pipeline_model, meta_props=metadata, experiment_metadata=experiment_metadata)

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction Pipeline_1 model'
            }
            stored_model_details = await client._models.astore(model="Pipeline_1", meta_props=metadata, training_id = training_id)

        Example of storing a prompt tuned model:

        .. code-block:: python

            stored_model_details = await client._models.astore(training_id = prompt_tuning_run_id)

        Example of storing a custom foundation model:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.0')

            metadata = {
                client.repository.ModelMetaNames.NAME: 'custom FM asset',
                client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0
            }
            stored_model_details = await client._models.astore(model='mistralai/Mistral-7B-Instruct-v0.2', meta_props=metadata)

        """

        # Import here to avoid circular import
        from ibm_watsonx_ai.foundation_models.utils.utils import (  # pylint: disable=import-outside-toplevel
            ais_training_prompt_tuning,
        )

        if (
            self._client.default_space_id is None
            and self._client.default_project_id is None
        ):
            raise WMLClientError(
                "It is mandatory is set the space or project. "
                "Use client.set.default_space(<SPACE_ID>) to set the space "
                "or client.set.default_project(<PROJECT_ID>)."
            )

        if isinstance(meta_props, dict) and (
            "project" in meta_props or "space" in meta_props
        ):
            raise WMLClientError(
                "'project' (MetaNames.PROJECT_ID) and 'space' (MetaNames.SPACE_ID) "
                "meta names are deprecated and considered as invalid. "
                "Instead use client.set.default_space(<SPACE_ID>) to set "
                "the space or client.set.default_project(<PROJECT_ID>)."
            )

        model_type = self._get_store_model_type(meta_props)

        is_prompt_tuned_training = await ais_training_prompt_tuning(
            training_id, api_client=self._client
        )

        Models._validate_type(
            meta_props,
            "meta_props",
            dict,
            mandatory=not is_prompt_tuned_training,
        )

        is_do_model = (
            (meta_props or {})
            .get(self.ConfigurationMetaNames.TYPE, "")
            .startswith("do-")
        )  # For DO model can be None, see #38648

        Models._validate_type(
            model,
            "model",
            object,
            mandatory=not (is_prompt_tuned_training or is_do_model),
        )

        meta_props = self._prepare_store_meta_props(meta_props)

        is_training_microservice = model is None or (
            isinstance(model, str) and "autoai_sdk" in model
        )

        # note: do not validate meta props when we have them from training microservice (always correct)
        if not is_training_microservice:
            if experiment_metadata or training_id:
                # note: if experiment_metadata are not None it means that the model is created from experiment,
                # and all required information are known from the experiment metadata and the origin
                Models._validate_type(meta_props, "meta_props", dict, True)
                Models._validate_type(meta_props["name"], "meta_props.name", str, True)
            else:
                self.ConfigurationMetaNames._validate(meta_props)

        framework_name = meta_props.get("frameworkName", "").lower()
        if version and framework_name in {"mllib", "wml"}:
            raise WMLClientError(
                "Unsupported framework name: '{}' for creating a model version".format(
                    framework_name
                )
            )

        if training_id and is_prompt_tuned_training:
            saved_model = await self._astore_prompt_tuning_model(
                training_id, meta_props
            )
        elif model is None:
            saved_model = await self._apublish_empty_model_asset(meta_props)
        elif model_type == "custom_model":
            saved_model = await self._astore_custom_foundation_model(model, meta_props)
        elif model_type == "curated_model":
            saved_model = await self._astore_type_of_foundation_model(
                model, meta_props, "curated"
            )
        elif model_type == "base_foundation_model":
            saved_model = await self._astore_type_of_foundation_model(
                model, meta_props, "base"
            )
        elif isinstance(model, str):
            saved_model = await self._astore_from_str(
                model,
                meta_props,
                training_data,
                training_target,
                version,
                artifactid,
                feature_names,
                label_column_names,
                subtrainingId,
                round_number,
                experiment_metadata,
                training_id,
            )
        else:
            saved_model = await self._astore_from_object(
                model,
                meta_props,
                training_data,
                training_target,
                pipeline,
                version,
                feature_names,
                label_column_names,
                experiment_metadata,
                training_id,
            )

        if warning_list := get_from_json(saved_model, ["system", "warnings"]):
            message = warning_list[0].get("message", None)
            print("Note: Warnings!! : ", message)

        return saved_model

    def _update_model_content(
        self, model_id: str, updated_details: dict[str, Any], update_model: Any
    ) -> None:
        model = copy.copy(update_model)
        model_type = updated_details["entity"]["type"]

        model_filepath = model

        if "scikit-learn_" in model_type or "mllib_" in model_type:
            meta_props = updated_details["entity"]
            meta_data = MetaProps(meta_props)
            name = updated_details["metadata"]["name"]
            model_artifact = MLRepositoryArtifact(
                update_model, name=name, meta_props=meta_data, training_data=None
            )
            model_artifact.uid = model_id
            query_params = self._client._params()
            query_params["content_format"] = "native"
            self._client.repository._ml_repository_client.models.upload_content(
                model_artifact, query_param=query_params, no_delete=True
            )
            return

        if (
            (os.path.sep in update_model)
            or os.path.isfile(update_model)
            or os.path.isdir(update_model)
        ):
            if not os.path.isfile(update_model) and not os.path.isdir(update_model):
                raise WMLClientError(
                    f"Invalid path: neither file nor directory exists under this path: '{model}'."
                )

        if os.path.isdir(model):
            if "tensorflow" in model_type:
                # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                if os.path.basename(model) == "":
                    model = os.path.dirname(update_model)
                filename = os.path.basename(update_model) + ".tar.gz"
                current_dir = os.getcwd()
                os.chdir(model)
                target_path = os.path.dirname(model)

                with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
                    tar.add(".")

                os.chdir(current_dir)
                path_to_archive = str(os.path.join(target_path, filename))
            else:
                if "caffe" in model_type:
                    raise WMLClientError(
                        f"Invalid model file path specified for: '{model_type}'."
                    )
                path_to_archive = load_model_from_directory(model_type, model)
        elif model_filepath.endswith(".xml"):
            path_to_archive = model_filepath
        elif model_filepath.endswith(".pmml"):
            raise WMLClientError(
                "The file name has an unsupported extension. Rename the file with a .xml extension."
            )
        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            path_to_archive = model_filepath
        else:
            raise WMLClientError(
                f"Saving trained model in repository failed. '{model_filepath}' file does not have valid format"
            )

        url = self._client._href_definitions.get_published_model_content_href(model_id)
        with open(path_to_archive, "rb") as file:
            params = self._client._params()

            if path_to_archive.endswith(".xml"):
                response = requests.put(
                    url,
                    data=file,
                    params=params,
                    headers=self._client._get_headers(content_type="application/xml"),
                )
            elif model_type.startswith("wml-hybrid_0"):
                response = self._upload_autoai_model_content(file, url, params)
            else:
                params["content_format"] = "native"
                response = requests.put(
                    url,
                    data=file,
                    params=params,
                    headers=self._client._get_headers(
                        content_type="application/octet-stream"
                    ),
                )

        self._handle_response(201, "uploading model content", response, False)

    def update(
        self,
        model_id: str | None = None,
        meta_props: dict | None = None,
        update_model: MLModelType = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Update an existing model.

        :param model_id: ID of model to be updated
        :type model_id: str

        :param meta_props: new set of meta_props to be updated
        :type meta_props: dict, optional

        :param update_model: archived model content file or path to directory that contains the archived model file
            that needs to be changed for the specific model_id
        :type update_model: object or model, optional

        :return: updated metadata of the model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client._models.update(model_id, update_model=updated_content)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        Models._validate_type(model_id, "model_id", str, False)
        model_id = cast(str, model_id)
        Models._validate_type(meta_props, "meta_props", dict, True)

        if meta_props is None:
            return self.get_details(model_id)

        self._validate_type(meta_props, "meta_props", dict, True)

        response = requests.get(
            self._client._href_definitions.get_published_model_href(model_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if response.status_code == 404:
            raise WMLClientError(
                "Invalid input. Unable to get the details of model_id provided."
            )

        if response.status_code != 200:
            raise ApiRequestFailure("Failure during getting model to update.", response)

        details = self._handle_response(200, "Get model details", response)
        model_type = details["entity"]["type"]

        # update the content path for the Auto-ai model.
        if (
            model_type == "wml-hybrid_0.1"
            and update_model is not None
            and not update_model.endswith(".zip")
        ):
            # The only supported format is a zip file containing `pipeline-model.json`
            # and pickled model compressed to tar.gz format.
            raise WMLClientError(
                "Invalid model content. The model content file should be zip archive containing "
                f'".pickle.tar.gz" file or "pipline-model.json", for the model type\'{model_type}\'.'
            )

        # with validation should be somewhere else, on the beginning, but when patch will be possible
        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, meta_props, with_validation=True
        )

        response_patch = requests.patch(
            self._client._href_definitions.get_published_model_href(model_id),
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        updated_details = self._handle_response(
            200, "model version patch", response_patch
        )

        if update_model is not None:
            self._update_model_content(model_id, details, update_model)

        return updated_details

    async def _aupdate_model_content(
        self, model_id: str, updated_details: dict[str, Any], update_model: Any
    ) -> None:
        model = copy.copy(update_model)
        model_type = updated_details["entity"]["type"]

        model_filepath = model

        if "scikit-learn_" in model_type or "mllib_" in model_type:
            meta_props = updated_details["entity"]
            meta_data = MetaProps(meta_props)
            name = updated_details["metadata"]["name"]
            model_artifact = MLRepositoryArtifact(
                update_model, name=name, meta_props=meta_data, training_data=None
            )
            model_artifact.uid = model_id
            query_params = self._client._params()
            query_params["content_format"] = "native"
            self._client.repository._ml_repository_client.models.upload_content(
                model_artifact, query_param=query_params, no_delete=True
            )
            return

        if (
            (os.path.sep in update_model)
            or os.path.isfile(update_model)
            or os.path.isdir(update_model)
        ):
            if not os.path.isfile(update_model) and not os.path.isdir(update_model):
                raise WMLClientError(
                    f"Invalid path: neither file nor directory exists under this path: '{model}'."
                )

        if os.path.isdir(model):
            if "tensorflow" in model_type:
                # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                if os.path.basename(model) == "":
                    model = os.path.dirname(update_model)
                filename = os.path.basename(update_model) + ".tar.gz"
                current_dir = os.getcwd()
                os.chdir(model)
                target_path = os.path.dirname(model)

                with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
                    tar.add(".")

                os.chdir(current_dir)
                path_to_archive = str(os.path.join(target_path, filename))
            else:
                if "caffe" in model_type:
                    raise WMLClientError(
                        f"Invalid model file path specified for: '{model_type}'."
                    )
                path_to_archive = load_model_from_directory(model_type, model)
        elif model_filepath.endswith(".xml"):
            path_to_archive = model_filepath
        elif model_filepath.endswith(".pmml"):
            raise WMLClientError(
                "The file name has an unsupported extension. Rename the file with a .xml extension."
            )
        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            path_to_archive = model_filepath
        else:
            raise WMLClientError(
                f"Saving trained model in repository failed. '{model_filepath}' file does not have valid format"
            )

        url = self._client._href_definitions.get_published_model_content_href(model_id)
        params = self._client._params()

        if path_to_archive.endswith(".xml"):
            response = await self._client.async_httpx_client.put(
                url,
                content=AsyncFileReader(path_to_archive),
                params=params,
                headers=await self._client._aget_headers(
                    content_type="application/xml"
                ),
            )
        elif model_type.startswith("wml-hybrid_0"):
            response = self._upload_autoai_model_content(path_to_archive, url, params)
        else:
            params["content_format"] = "native"
            response = await self._client.async_httpx_client.put(
                url,
                content=AsyncFileReader(path_to_archive),
                params=params,
                headers=await self._client._aget_headers(
                    content_type="application/octet-stream"
                ),
            )

        self._handle_response(201, "uploading model content", response, False)

    async def aupdate(
        self,
        model_id: str,
        meta_props: dict | None = None,
        update_model: MLModelType = None,
    ) -> dict[str, Any]:
        """Update an existing model asynchronously.

        :param model_id: ID of model to be updated
        :type model_id: str

        :param meta_props: new set of meta_props to be updated
        :type meta_props: dict, optional

        :param update_model: archived model content file or path to directory that contains the archived model file
            that needs to be changed for the specific model_id
        :type update_model: object or model, optional

        :return: updated metadata of the model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = await client._models.aupdate(model_id, update_model=updated_content)
        """

        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(meta_props, "meta_props", dict, False)

        if meta_props is None:
            return await self.aget_details(model_id)

        response = await self._client.async_httpx_client.get(
            self._client._href_definitions.get_published_model_href(model_id),
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        if response.status_code == 404:
            raise WMLClientError(
                "Invalid input. Unable to get the details of model_id provided."
            )

        if response.status_code != 200:
            raise ApiRequestFailure("Failure during getting model to update.", response)

        details = self._handle_response(200, "Get model details", response)
        model_type = details["entity"]["type"]

        # update the content path for the Auto-ai model.
        if (
            model_type == "wml-hybrid_0.1"
            and update_model is not None
            and not update_model.endswith(".zip")
        ):
            # The only supported format is a zip file containing `pipeline-model.json`
            # and pickled model compressed to tar.gz format.
            raise WMLClientError(
                "Invalid model content. The model content file should be zip archive containing "
                f'".pickle.tar.gz" file or "pipline-model.json", for the model type\'{model_type}\'.'
            )

        # with validation should be somewhere else, on the beginning, but when patch will be possible
        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details, meta_props, with_validation=True
        )

        response_patch = await self._client.async_httpx_client.patch(
            self._client._href_definitions.get_published_model_href(model_id),
            json=patch_payload,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        updated_details = self._handle_response(
            200, "model version patch", response_patch
        )

        if update_model is not None:
            await self._aupdate_model_content(model_id, details, update_model)

        return updated_details

    def load(self, artifact_id: str | None, **kwargs) -> Any:
        """Load a model from the repository to object in a local environment.

        .. note::
            The use of the load() method is restricted and not permitted for AutoAI models.

        :param artifact_id: ID of the stored model
        :type artifact_id: str

        :return: trained model
        :rtype: object

        **Example:**

        .. code-block:: python

            model = client._models.load(model_id)
        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        artifact_id = cast(str, artifact_id)
        Models._validate_type(artifact_id, "artifact_id", str, True)

        # check if this is tensorflow 2.x model type
        model_details = self.get_details(artifact_id)
        model_type = get_from_json(model_details, ["entity", "type"], "")

        if "wml-hybrid" in model_type:
            raise WMLClientError(
                "The use of the load() method is restricted and not permitted for AutoAI models."
            )

        if model_type.startswith("tensorflow_2."):
            return self._tf2x_load_model_instance(artifact_id)

        try:
            # Cloud Convergence: CHK IF THIS CONDITION IS CORRECT since loaded_model
            # functionality below
            if (
                self._client.default_space_id is None
                and self._client.default_project_id is None
            ):
                raise WMLClientError(
                    "It is mandatory is set the space or project. \
                    Use client.set.default_space(<SPACE_ID>) to set the space or client.set.default_project(<PROJECT_ID>)."
                )

            query_param = self._client._params()
            loaded_model = self._client.repository._ml_repository_client.models._get_v4_cloud_model(
                artifact_id, query_param=query_param
            )
            loaded_model_instance = loaded_model.model_instance()

            self._logger.info(
                "Successfully loaded artifact with artifact_id: %s", artifact_id
            )
            return loaded_model_instance
        except Exception as e:
            raise WMLClientError(
                f"Loading model with artifact_id: '{artifact_id}' failed.", e
            )

    async def aload(self, artifact_id: str) -> Any:
        """Load a model from the repository to object in a local environment asynchronously.

        :param artifact_id: ID of the stored model
        :type artifact_id: str

        :return: trained model
        :rtype: object

        **Example:**

        .. code-block:: python

            model = await client._models.aload(model_id)
        """
        Models._validate_type(artifact_id, "artifact_id", str, True)

        # check if this is tensorflow 2.x model type
        model_details = await self.aget_details(artifact_id)
        model_type = get_from_json(model_details, ["entity", "type"], "")

        if "wml-hybrid" in model_type:
            raise WMLClientError(
                "The use of the load() method is restricted and not permitted for AutoAI models."
            )

        if model_type.startswith("tensorflow_2."):
            return await self._atf2x_load_model_instance(artifact_id)

        try:
            # Cloud Convergence: CHK IF THIS CONDITION IS CORRECT since loaded_model
            # functionality below
            if (
                self._client.default_space_id is None
                and self._client.default_project_id is None
            ):
                raise WMLClientError(
                    "It is mandatory is set the space or project. \
                    Use client.set.default_space(<SPACE_ID>) to set the space or client.set.default_project(<PROJECT_ID>)."
                )

            query_param = self._client._params()
            loaded_model = self._client.repository._ml_repository_client.models._get_v4_cloud_model(
                artifact_id, query_param=query_param
            )
            loaded_model_instance = loaded_model.model_instance()

            self._logger.info(
                "Successfully loaded artifact with artifact_id: %s", artifact_id
            )
            return loaded_model_instance
        except Exception as e:
            raise WMLClientError(
                f"Loading model with artifact_id: '{artifact_id}' failed.", e
            )

    def _download_auto_ai_model_content(
        self, model_id: str, content_url: str, filename: str
    ) -> None:
        with zipfile.ZipFile(filename, "w") as zip_file:
            pipeline_model_file = "pipeline-model.json"

            with open(pipeline_model_file, "wb") as f:
                response = requests.get(
                    content_url,
                    params=self._client._params() | {"content_format": "native"},
                    headers=self._client._get_headers(),
                    stream=True,
                )

                if response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during downloading model.", response
                    )

                self._logger.info(
                    "Successfully downloaded artifact pipeline_model.json artifact_url: %s",
                    content_url,
                )
                f.write(response.content)

            zip_file.write(pipeline_model_file)
            mfilename = f"model_{model_id}.pickle.tar.gz"

            with open(mfilename, "wb") as f:
                response = requests.get(
                    content_url,
                    params=self._client._params() | {"content_format": "pipeline-node"},
                    headers=self._client._get_headers(),
                    stream=True,
                )

                if response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during downloading model.", response
                    )

                f.write(response.content)
                self._logger.info(
                    "Successfully downloaded artifact with artifact_url: %s",
                    content_url,
                )

            zip_file.write(mfilename)

    def download(
        self,
        model_id: str | None,
        filename: str = "downloaded_model.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
        **kwargs,
    ) -> str:
        """Download a model from the repository to local file.

        :param model_id: ID of the stored model
        :type model_id: str

        :param filename: name of local file to be created
        :type filename: str, optional

        :param rev_id: ID of the revision
        :type rev_id: str, optional

        :param format: format of the content
        :type format: str, optional

        **Example:**

        .. code-block:: python

            client._models.download(model_id, 'my_model.tar.gz')
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", True)

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(filename, "filename", str, True)

        if is_json := filename.endswith(".json"):
            json_filename = filename
            filename = f"tmp_{uuid.uuid4()}.tar.gz"

        artifact_url = self._client._href_definitions.get_model_last_version_href(
            model_id
        )

        try:
            model_get_response = requests.get(
                self._client._href_definitions.get_published_model_href(model_id),
                params=self._client._params(),
                headers=self._client._get_headers(),
            )

            model_details = self._handle_response(200, "get model", model_get_response)

            params = self._client._params()
            if rev_id is not None:
                params["revision_id"] = rev_id

            model_type = model_details["entity"]["type"]
            if (
                model_type.startswith("keras_")
                or model_type.startswith("scikit-learn_")
                or model_type.startswith("xgboost_")
            ) and format is not None:
                Models._validate_type(format, "format", str, False)
                if str(format).upper() == "COREML":
                    params["content_format"] = "coreml"
                else:
                    params["content_format"] = "native"
            else:
                params["content_format"] = "native"

            artifact_content_url = (
                self._client._href_definitions.get_model_download_href(model_id)
            )

            if model_details["entity"]["type"] == "wml-hybrid_0.1":
                self._download_auto_ai_model_content(
                    model_id, artifact_content_url, filename
                )
                print(f"Successfully saved model content to file: '{filename}'")
                return os.getcwd() + "/" + filename

            response = requests.get(
                artifact_content_url,
                params=params,
                headers=self._client._get_headers(),
                stream=True,
            )

            if response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading model"), response
                )

            downloaded_model = response.content
            self._logger.info(
                "Successfully downloaded artifact with artifact_url: %s", artifact_url
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            if artifact_url is not None:
                raise WMLClientError(
                    f"Downloading model with artifact_url: '{artifact_url}' failed.",
                    e,
                )
            else:
                raise WMLClientError("Downloading model failed.", e)
        finally:
            if is_json:
                try:
                    os.remove(filename)
                except Exception:
                    pass

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)

            if is_json:
                with tarfile.open(filename, "r:gz") as tar:
                    file_name = tar.getnames()[0]
                    if not file_name.endswith(".json"):
                        raise WMLClientError("Downloaded model is not json.")
                    tar.extractall()

                os.rename(file_name, json_filename)
                os.remove(filename)

                filename = json_filename

            print(f"Successfully saved model content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving model with artifact_url: '{filename}' failed.", e
            )

    async def _adownload_auto_ai_model_content(
        self, model_id: str, content_url: str, filename: str
    ) -> None:
        with zipfile.ZipFile(filename, "w") as zip_file:
            pipeline_model_file = "pipeline-model.json"

            with open(pipeline_model_file, "wb") as f:
                async with self._client.async_httpx_client.stream(
                    method="GET",
                    url=content_url,
                    params=self._client._params() | {"content_format": "native"},
                    headers=await self._client._aget_headers(),
                ) as response:
                    if response.status_code != 200:
                        raise ApiRequestFailure(
                            "Failure during downloading model.", response
                        )

                    self._logger.info(
                        "Successfully downloaded artifact pipeline_model.json artifact_url: %s",
                        content_url,
                    )
                    f.write(await response.aread())

            zip_file.write(pipeline_model_file)
            mfilename = f"model_{model_id}.pickle.tar.gz"

            with open(mfilename, "wb") as f:
                async with self._client.async_httpx_client.stream(
                    method="GET",
                    url=content_url,
                    params=self._client._params() | {"content_format": "pipeline-node"},
                    headers=await self._client._aget_headers(),
                ) as response:
                    if response.status_code != 200:
                        raise ApiRequestFailure(
                            "Failure during downloading model.", response
                        )

                    self._logger.info(
                        "Successfully downloaded artifact with artifact_url: %s",
                        content_url,
                    )

                    f.write(await response.aread())

            zip_file.write(mfilename)

    async def adownload(
        self,
        model_id: str,
        filename: str = "downloaded_model.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
    ) -> str:
        """Download a model from the repository to local file asynchronously.

        :param model_id: ID of the stored model
        :type model_id: str

        :param filename: name of local file to be created
        :type filename: str, optional

        :param rev_id: ID of the revision
        :type rev_id: str, optional

        :param format: format of the content
        :type format: str, optional

        **Example:**

        .. code-block:: python

            await client._models.adownload(model_id, 'my_model.tar.gz')
        """

        if os.path.isfile(filename):
            raise WMLClientError(f"File with name: '{filename}' already exists.")

        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(filename, "filename", str, True)

        if is_json := filename.endswith(".json"):
            json_filename = filename
            filename = f"tmp_{uuid.uuid4()}.tar.gz"
        else:
            json_filename = ""

        artifact_url = self._client._href_definitions.get_model_last_version_href(
            model_id
        )

        try:
            model_get_response = await self._client.async_httpx_client.get(
                self._client._href_definitions.get_published_model_href(model_id),
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )

            model_details = self._handle_response(200, "get model", model_get_response)
            params = self._client._params()
            if rev_id is not None:
                params["revision_id"] = rev_id

            model_type = model_details["entity"]["type"]
            if (
                model_type.startswith("keras_")
                or model_type.startswith("scikit-learn_")
                or model_type.startswith("xgboost_")
            ) and format is not None:
                Models._validate_type(format, "format", str, False)
                if str(format).upper() == "COREML":
                    params["content_format"] = "coreml"
                else:
                    params["content_format"] = "native"
            else:
                params["content_format"] = "native"

            artifact_content_url = (
                self._client._href_definitions.get_model_download_href(model_id)
            )

            if model_details["entity"]["type"] == "wml-hybrid_0.1":
                await self._adownload_auto_ai_model_content(
                    model_id, artifact_content_url, filename
                )
                print(f"Successfully saved model content to file: '{filename}'")
                return os.getcwd() + "/" + filename

            async with self._client.async_httpx_client.stream(
                method="GET",
                url=artifact_content_url,
                params=params,
                headers=await self._client._aget_headers(),
            ) as response:
                if response.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading model"), response
                    )

                downloaded_model = await response.aread()
                self._logger.info(
                    "Successfully downloaded artifact with artifact_url: %s",
                    artifact_url,
                )
        except WMLClientError as e:
            raise e
        except Exception as e:
            if artifact_url is not None:
                raise WMLClientError(
                    f"Downloading model with artifact_url: '{artifact_url}' failed.",
                    e,
                )
            else:
                raise WMLClientError("Downloading model failed.", e)
        finally:
            if is_json:
                try:
                    os.remove(filename)
                except Exception:
                    pass

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)

            if is_json:
                with tarfile.open(filename, "r:gz") as tar:
                    file_name = tar.getnames()[0]
                    if not file_name.endswith(".json"):
                        raise WMLClientError("Downloaded model is not json.")
                    tar.extractall()

                os.rename(file_name, json_filename)
                os.remove(filename)

                filename = json_filename

            print(f"Successfully saved model content to file: '{filename}'")
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                f"Saving model with artifact_url: '{filename}' failed.", e
            )

    def delete(
        self, model_id: str | None = None, force: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Delete a model from the repository.

        :param model_id: ID of the stored model
        :type model_id: str

        :param force: if True, the delete operation will proceed even when the model deployment exists, defaults to False
        :type force: bool, optional

        **Example:**

        .. code-block:: python

            client._models.delete(model_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        Models._validate_type(model_id, "model_id", str, False)

        if not force and self._if_deployment_exist_for_asset(model_id):
            raise WMLClientError(
                "Cannot delete model that has existing deployments. Please delete all associated deployments and try again"
            )

        model_endpoint = self._client._href_definitions.get_published_model_href(
            model_id
        )

        self._logger.debug("Deletion artifact model endpoint: %s", model_endpoint)

        response = requests.delete(
            model_endpoint,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "model deletion", response, False)

    async def adelete(self, model_id: str, force: bool = False) -> dict[str, Any]:
        """Delete a model from the repository asynchronously.

        :param model_id: ID of the stored model
        :type model_id: str

        :param force: if True, the delete operation will proceed even when the model deployment exists, defaults to False
        :type force: bool, optional

        **Example:**

        .. code-block:: python

            await client._models.adelete(model_id)
        """
        Models._validate_type(model_id, "model_id", str, True)

        if not force and await self._aif_deployment_exist_for_asset(model_id):
            raise WMLClientError(
                "Cannot delete model that has existing deployments. Please delete all associated deployments and try again"
            )

        model_endpoint = self._client._href_definitions.get_published_model_href(
            model_id
        )

        self._logger.debug("Deletion artifact model endpoint: %s", model_endpoint)

        response = await self._client.async_httpx_client.delete(
            model_endpoint,
            params=self._client._params(),
            headers=await self._client._aget_headers(),
        )

        return self._handle_response(204, "model deletion", response, False)

    @overload
    def get_details(
        self,
        model_id: str = "",
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]: ...

    @overload
    def get_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any] | Generator: ...

    def get_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any] | Generator:
        """Get metadata of stored models. If neither model ID nor model name is specified,
        the metadata of all models is returned.
        If only model name is specified, metadata of models with the name is returned (if any).

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `model_id` is None
        :type spec_state: SpecStates, optional

        :param model_name: name of the stored model, definition, or pipeline, can be used only when `model_id` is None
        :type model_name: str, optional

        :return: metadata of the stored model(s)
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state` may break set `limit`,
            returning less records than stated by set `limit`.

        **Example:**

        .. code-block:: python

            model_details = client._models.get_details(model_id)
            models_details = client._models.get_details(model_name='Sample_model')
            models_details = client._models.get_details()
            models_details = client._models.get_details(limit=100)
            models_details = client._models.get_details(limit=100, get_all=True)
            models_details = []
            for entry in client._models.get_details(limit=100, asynchronous=True, get_all=True):
                models_details.extend(entry)

        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model", True)
        if limit and spec_state:
            spec_state_setting_warning = (
                "Warning: In current implementation setting `spec_state` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )
            warn(spec_state_setting_warning, category=DeprecationWarning)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, False)
        Models._validate_type(limit, "limit", int, False)

        url = self._client._href_definitions.get_published_models_href()

        if model_id is None:
            if spec_state:
                filter_func = self._get_filter_func_by_spec_ids(
                    self._get_and_cache_spec_ids_for_state(spec_state)
                )
            elif model_name:
                filter_func = self._get_filter_func_by_artifact_name(model_name)
            else:
                filter_func = None

            return self._get_artifact_details(
                url,
                model_id,
                limit,
                "models",
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )

        return self._get_artifact_details(url, model_id, limit, "models")

    @overload
    async def aget_details(
        self,
        model_id: str = "",
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]: ...

    @overload
    async def aget_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any] | AsyncGenerator: ...

    async def aget_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any] | AsyncGenerator:
        """Get metadata of stored models asynchronously. If neither model ID nor model name is specified,
        the metadata of all models is returned.
        If only model name is specified, metadata of models with the name is returned (if any).

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `model_id` is None
        :type spec_state: SpecStates, optional

        :param model_name: name of the stored model, definition, or pipeline, can be used only when `model_id` is None
        :type model_name: str, optional

        :return: metadata of the stored model(s)
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state` may break set `limit`,
            returning less records than stated by set `limit`.

        **Example:**

        .. code-block:: python

            model_details = await client._models.aget_details(model_id)
            models_details = await client._models.aget_details(model_name='Sample_model')
            models_details = await client._models.aget_details()
            models_details = await client._models.aget_details(limit=100)
            models_details = await client._models.aget_details(limit=100, get_all=True)
            models_details = []
            async for entry in await client._models.aget_details(limit=100, asynchronous=True, get_all=True):
                models_details.extend(entry)

        """

        if limit and spec_state:
            spec_state_setting_warning = (
                "Warning: In current implementation setting `spec_state` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )
            warn(spec_state_setting_warning, category=DeprecationWarning)

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, False)
        Models._validate_type(limit, "limit", int, False)

        url = self._client._href_definitions.get_published_models_href()

        if model_id is None:
            if spec_state:
                filter_func = self._get_filter_func_by_spec_ids(
                    self._get_and_cache_spec_ids_for_state(spec_state)
                )
            elif model_name:
                filter_func = self._get_filter_func_by_artifact_name(model_name)
            else:
                filter_func = None

            return await self._aget_artifact_details(
                url,
                model_id,
                limit,
                "models",
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )

        return await self._aget_artifact_details(url, model_id, limit, "models")

    @staticmethod
    def get_href(model_details: dict[str, Any]) -> str:
        """Get the URL of a stored model.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: URL of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_url = client._models.get_href(model_details)
        """

        Models._validate_type(model_details, "model_details", object, True)

        if "asset_id" in model_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "href"]
            )

        if "id" not in model_details["metadata"]:
            Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "href"]
            )

        model_id = WMLResource._get_required_element_from_dict(
            model_details, "model_details", ["metadata", "id"]
        )
        return f"/ml/v4/models/{model_id}"

    @staticmethod
    def get_uid(model_details: dict[str, Any]) -> str:
        """Get the UID of a stored model.

        *Deprecated:* Use ``get_id(model_details)`` instead.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: UID of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_uid = client._models.get_uid(model_details)
        """
        get_uid_method_deprecated_warning = (
            "This method is deprecated, please use Models.get_id(model_details) instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)
        return Models.get_id(model_details)

    @staticmethod
    def get_id(model_details: dict[str, Any]) -> str:
        """Get the ID of a stored model.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: ID of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_id = client._models.get_id(model_details)
        """
        Models._validate_type(model_details, "model_details", object, True)

        if "asset_id" in model_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "asset_id"]
            )

        if "id" not in model_details["metadata"]:
            Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "guid"]
            )

        return WMLResource._get_required_element_from_dict(
            model_details, "model_details", ["metadata", "id"]
        )

    def _process_resources_for_list(
        self, model_resources: dict, limit: int | None
    ) -> pandas.DataFrame:
        model_resources = model_resources["resources"]

        model_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
                m["entity"]["type"],
                self._client.software_specifications._get_state(m),
                self._client.software_specifications._get_replacement(m),
            )
            for m in model_resources
        ]

        return self._list(
            model_values,
            ["ID", "NAME", "CREATED", "TYPE", "SPEC_STATE", "SPEC_REPLACEMENT"],
            limit,
        )

    def list(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> pandas.DataFrame | Generator:
        """List stored models in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: pandas.DataFrame with listed models or generator if `asynchronous` is set to True
        :rtype: pandas.DataFrame | Generator

        **Example:**

        .. code-block:: python

            client._models.list()
            client._models.list(limit=100)
            client._models.list(limit=100, get_all=True)
            [entry for entry in client._models.list(limit=100, asynchronous=True, get_all=True)]
        """

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        if asynchronous:
            return (
                self._process_resources_for_list(model_resources, limit)  # type: ignore[arg-type]
                for model_resources in self.get_details(
                    limit=limit, asynchronous=asynchronous, get_all=get_all
                )
            )

        model_resources = self.get_details(limit=limit, get_all=get_all)
        return self._process_resources_for_list(model_resources, limit)

    def create_revision(self, model_id: str | None = None, **kwargs) -> dict[str, Any]:
        """Create a revision for a given model ID.

        :param model_id: ID of the stored model
        :type model_id: str

        :return: revised metadata of the stored model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client._models.create_revision(model_id)
        """
        # For CP4D, check if either space or project ID is set
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, False)

        return self._create_revision_artifact(
            self._client._href_definitions.get_published_models_href(),
            model_id,
            "models",
        )

    async def acreate_revision(self, model_id: str) -> dict[str, Any]:
        """Create a revision for a given model ID asynchronously.

        :param model_id: ID of the stored model
        :type model_id: str

        :return: revised metadata of the stored model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = await client._models.acreate_revision(model_id)
        """

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, True)

        return await self._acreate_revision_artifact(
            self._client._href_definitions.get_published_models_href(),
            model_id,
            "models",
        )

    def list_revisions(
        self, model_id: str | None = None, limit: int | None = None, **kwargs
    ) -> pandas.DataFrame:
        """Print all revisions for the given model ID in a table format.

        :param model_id: unique ID of the stored model
        :type model_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._models.list_revisions(model_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Models._validate_type(model_id, "model_id", str, True)

        model_resources = self._get_artifact_details(
            self._client._href_definitions.get_model_last_version_href(model_id),
            "revisions",
            None,
            "model revisions",
            _all=self._should_get_all_values(limit),
        )["resources"]

        model_values = [
            (m["metadata"]["rev"], m["metadata"]["name"], m["metadata"]["created_at"])
            for m in model_resources
        ]

        return self._list(model_values, ["REV", "NAME", "CREATED"], limit)

    def get_revision_details(
        self, model_id: str | None = None, rev_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Get metadata of a stored model's specific revision.

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str

        :param rev_id: unique ID of the stored model revision
        :type rev_id: str

        :return: metadata of the stored model(s)
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client._models.get_revision_details(model_id, rev_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(rev_id, "rev_id", str, True)

        return self._get_with_or_without_limit(
            self._client._href_definitions.get_published_model_href(model_id),
            limit=None,
            op_name="getting revision details",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    async def aget_revision_details(self, model_id: str, rev_id: str) -> dict[str, Any]:
        """Get metadata of a stored model's specific revision asynchronously.

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str

        :param rev_id: unique ID of the stored model revision
        :type rev_id: str

        :return: metadata of the stored model(s)
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = await client._models.aget_revision_details(model_id, rev_id)
        """

        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(rev_id, "rev_id", str, True)

        return await self._aget_with_or_without_limit(
            self._client._href_definitions.get_published_model_href(model_id),
            limit=None,
            op_name="getting revision details",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def promote(
        self, model_id: str, source_project_id: str, target_space_id: str
    ) -> dict[str, Any] | None:
        """Promote a model from a project to space. Supported only for IBM Cloud Pak® for Data.

        *Deprecated:* Use `client.spaces.promote(asset_id, source_project_id, target_space_id)` instead.
        """

        promote_model_method_deprecated_warning = (
            "Note: Function `client.repository.promote_model(model_id, source_project_id, target_space_id)` "
            "has been deprecated. Use `client.spaces.promote(asset_id, source_project_id, target_space_id)` instead."
        )
        warn(promote_model_method_deprecated_warning, category=DeprecationWarning)

        try:
            return self._client.spaces.promote(
                model_id, source_project_id, target_space_id
            )
        except PromotionFailed as e:
            raise ModelPromotionFailed(
                e.project_id, e.space_id, e.promotion_response, str(e.reason)
            )
