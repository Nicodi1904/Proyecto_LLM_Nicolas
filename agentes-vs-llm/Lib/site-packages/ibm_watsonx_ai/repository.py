#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
    TypeAlias,
    cast,
)
from warnings import warn

import httpx
import requests as _requests

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.ai_services import AIServices
from ibm_watsonx_ai.experiments import Experiments
from ibm_watsonx_ai.functions import Functions
from ibm_watsonx_ai.libs.repo.mlrepositoryclient import MLRepositoryClient
from ibm_watsonx_ai.lifecycle import SpecStates
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import (
    AIServiceMetaNames,
    ExperimentMetaNames,
    FunctionMetaNames,
    ModelMetaNames,
    PipelineMetanames,
    RepositoryMemberMetaNames,
    SpacesMetaNames,
)
from ibm_watsonx_ai.models import Models
from ibm_watsonx_ai.pipelines import Pipelines
from ibm_watsonx_ai.utils import get_user_agent_header, inherited_docstring
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    import numpy
    import pandas
    import pyspark.sql

    from ibm_watsonx_ai import APIClient

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


class Repository(WMLResource):
    """Store and manage models, functions, spaces, pipelines, and experiments
    using the Watson Machine Learning Repository.

    To view ModelMetaNames, use:

    .. code-block:: python

        client.repository.ModelMetaNames.show()

    To view ExperimentMetaNames, use:

    .. code-block:: python

        client.repository.ExperimentMetaNames.show()

    To view FunctionMetaNames, use:

    .. code-block:: python

        client.repository.FunctionMetaNames.show()

    To view PipelineMetaNames, use:

    .. code-block:: python

        client.repository.PipelineMetaNames.show()

    To view AIServiceMetaNames, use:

    .. code-block:: python

        client.repository.AIServiceMetaNames.show()

    """

    @dataclass
    class ModelAssetTypes:
        """Data class with supported model asset types."""

        DO_DOCPLEX_20_1: str = "do-docplex_20.1"
        DO_OPL_20_1: str = "do-opl_20.1"
        DO_CPLEX_20_1: str = "do-cplex_20.1"
        DO_CPO_20_1: str = "do-cpo_20.1"
        DO_DOCPLEX_22_1: str = "do-docplex_22.1"
        DO_OPL_22_1: str = "do-opl_22.1"
        DO_CPLEX_22_1: str = "do-cplex_22.1"
        DO_CPO_22_1: str = "do-cpo_22.1"
        WML_HYBRID_0_1: str = "wml-hybrid_0.1"
        PMML_4_2_1: str = "pmml_4.2.1"
        PYTORCH_ONNX_1_12: str = "pytorch-onnx_1.12"
        PYTORCH_ONNX_RT22_2: str = "pytorch-onnx_rt22.2"
        PYTORCH_ONNX_2_0: str = "pytorch-onnx_2.0"
        PYTORCH_ONNX_RT23_1: str = "pytorch-onnx_rt23.1"
        SCIKIT_LEARN_1_1: str = "scikit-learn_1.1"
        MLLIB_3_3: str = "mllib_3.3"
        SPSS_MODELER_17_1: str = "spss-modeler_17.1"
        SPSS_MODELER_18_1: str = "spss-modeler_18.1"
        SPSS_MODELER_18_2: str = "spss-modeler_18.2"
        TENSORFLOW_2_9: str = "tensorflow_2.9"
        TENSORFLOW_RT22_2: str = "tensorflow_rt22.2"
        TENSORFLOW_2_12: str = "tensorflow_2.12"
        TENSORFLOW_RT23_1: str = "tensorflow_rt23.1"
        XGBOOST_1_6: str = "xgboost_1.6"
        PROMPT_TUNE_1_0: str = "prompt_tune_1.0"
        CUSTOM_FOUNDATION_MODEL_1_0: str = "custom_foundation_model_1.0"
        CURATED_FOUNDATION_MODEL_1_0: str = "curated_foundation_model_1.0"
        BASE_FOUNDATION_MODEL_1_0: str = "base_foundation_model_1.0"

    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)
        self._ml_repository_client: MLRepositoryClient | None = None

        self.ExperimentMetaNames = ExperimentMetaNames()
        self.FunctionMetaNames = FunctionMetaNames()
        self.PipelineMetaNames = PipelineMetanames()
        self.SpacesMetaNames = SpacesMetaNames()
        self.ModelMetaNames = ModelMetaNames()
        self.MemberMetaNames = RepositoryMemberMetaNames()
        self.AIServiceMetaNames = AIServiceMetaNames()

        # make sure that old repo client is aware of token changes
        self._client._auth_method._on_token_set = self._refresh_repo_client  # type: ignore
        self._client._auth_method._on_token_creation = self._refresh_repo_client  # type: ignore
        self._client._auth_method._on_token_refresh = self._refresh_repo_client  # type: ignore

        self._refresh_repo_client()

    def _refresh_repo_client(self) -> None:
        self._ml_repository_client = MLRepositoryClient(self._credentials.url)
        # this is refresh-not-triggering get of token from client, added here especially for extra short living tokens
        self._ml_repository_client.authorize_with_token(
            self._client._auth_method._token
        )
        self._ml_repository_client._add_header("User-Agent", get_user_agent_header())

    def _get_artifact_endpoints(self, artifact_id: str) -> dict[str, str]:
        return {
            "model": self._client._href_definitions.get_model_last_version_href(
                artifact_id
            ),
            "pipeline": self._client._href_definitions.get_pipeline_href(artifact_id),
            "experiment": self._client._href_definitions.get_experiment_href(
                artifact_id
            ),
            "function": self._client._href_definitions.get_function_href(artifact_id),
            "ai_service": self._client._href_definitions.get_ai_service_href(
                artifact_id
            ),
        }

    def _check_artifact_type(self, artifact_id: str) -> dict[str, bool]:
        self._validate_type(artifact_id, "artifact_id", str, True)

        endpoints = self._get_artifact_endpoints(artifact_id)

        response_by_artifact_type = {
            artifact: requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            for artifact, url in endpoints.items()
        }

        return {
            artifact: (response.status_code == 200)
            for artifact, response in response_by_artifact_type.items()
        }

    async def _acheck_artifact_type(self, artifact_id: str) -> dict[str, bool]:
        self._validate_type(artifact_id, "artifact_id", str, True)

        endpoints = self._get_artifact_endpoints(artifact_id)

        response_by_artifact_type = {
            artifact: await self._client.async_httpx_client.get(
                url,
                params=self._client._params(),
                headers=await self._client._aget_headers(),
            )
            for artifact, url in endpoints.items()
        }

        return {
            artifact: (response.status_code == 200)
            for artifact, response in response_by_artifact_type.items()
        }

    @inherited_docstring(
        Experiments.store, {"experiments.get_href": "repository.get_experiment_href"}
    )
    def store_experiment(self, meta_props: dict) -> dict:
        return self._client.experiments.store(meta_props)

    @inherited_docstring(
        Experiments.astore, {"experiments.get_href": "repository.get_experiment_href"}
    )
    async def astore_experiment(self, meta_props: dict) -> dict:
        return await self._client.experiments.astore(meta_props)

    @inherited_docstring(Pipelines.store)
    def store_pipeline(self, meta_props: dict) -> dict:
        return self._client.pipelines.store(meta_props)

    @inherited_docstring(Pipelines.astore)
    async def astore_pipeline(self, meta_props: dict) -> dict:
        return await self._client.pipelines.astore(meta_props)

    @inherited_docstring(Models.store, {"store()": "store_model()"})
    def store_model(
        self,
        model: str | object | None = None,
        meta_props: dict | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: object | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict | None = None,
        training_id: str | None = None,
    ) -> dict:
        return self._client._models.store(
            model=model,
            meta_props=meta_props,
            training_data=training_data,
            training_target=training_target,
            pipeline=pipeline,
            feature_names=feature_names,
            label_column_names=label_column_names,
            subtrainingId=subtrainingId,
            round_number=round_number,
            experiment_metadata=experiment_metadata,
            training_id=training_id,
        )

    @inherited_docstring(Models.astore, {"astore()": "astore_model()"})
    async def astore_model(
        self,
        model: str | object | None = None,
        meta_props: dict | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: object | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict | None = None,
        training_id: str | None = None,
    ) -> dict:
        return await self._client._models.astore(
            model=model,
            meta_props=meta_props,
            training_data=training_data,
            training_target=training_target,
            pipeline=pipeline,
            feature_names=feature_names,
            label_column_names=label_column_names,
            subtrainingId=subtrainingId,
            round_number=round_number,
            experiment_metadata=experiment_metadata,
            training_id=training_id,
        )

    def clone(
        self,
        artifact_id: str,
        space_id: str | None = None,
        action: str = "copy",
        rev_id: str | None = None,
    ) -> dict:
        raise WMLClientError(Messages.get_message(message_id="cloning_not_supported"))

    @inherited_docstring(Functions.store)
    def store_function(
        self, function: str | Callable, meta_props: str | dict[str, Any]
    ) -> dict:
        return self._client._functions.store(function, meta_props)

    @inherited_docstring(Functions.astore)
    async def astore_function(
        self, function: str | Callable, meta_props: str | dict[str, Any]
    ) -> dict:
        return await self._client._functions.astore(function, meta_props)

    @inherited_docstring(Models.create_revision)
    def create_model_revision(self, model_id: str | None = None, **kwargs: Any) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.create_revision(model_id=model_id)

    @inherited_docstring(Models.acreate_revision)
    async def acreate_model_revision(self, model_id: str) -> dict:
        return await self._client._models.acreate_revision(model_id=model_id)

    @inherited_docstring(Pipelines.create_revision)
    def create_pipeline_revision(
        self, pipeline_id: str | None = None, **kwargs: Any
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.create_revision(pipeline_id=pipeline_id)

    @inherited_docstring(Pipelines.acreate_revision)
    async def acreate_pipeline_revision(self, pipeline_id: str) -> dict:
        return await self._client.pipelines.acreate_revision(pipeline_id=pipeline_id)

    @inherited_docstring(Functions.create_revision)
    def create_function_revision(
        self, function_id: str | None = None, **kwargs: Any
    ) -> dict:
        return self._client._functions.create_revision(
            function_id=function_id, **kwargs
        )

    @inherited_docstring(Functions.acreate_revision)
    async def acreate_function_revision(self, function_id: str) -> dict:
        return await self._client._functions.acreate_revision(function_id=function_id)

    @inherited_docstring(Experiments.create_revision)
    def create_experiment_revision(self, experiment_id: str) -> dict:
        return self._client.experiments.create_revision(experiment_id=experiment_id)

    @inherited_docstring(Experiments.acreate_revision)
    async def acreate_experiment_revision(self, experiment_id: str) -> dict:
        return await self._client.experiments.acreate_revision(
            experiment_id=experiment_id
        )

    @inherited_docstring(Models.update, {"meta_props": "updated_meta_props"})
    def update_model(
        self,
        model_id: str | None = None,
        updated_meta_props: dict | None = None,
        update_model: Any | None = None,
        **kwargs: Any,
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.update(model_id, updated_meta_props, update_model)

    @inherited_docstring(Models.aupdate, {"meta_props": "updated_meta_props"})
    async def aupdate_model(
        self,
        model_id: str,
        updated_meta_props: dict | None = None,
        update_model: Any | None = None,
    ) -> dict:
        return await self._client._models.aupdate(
            model_id, updated_meta_props, update_model
        )

    @inherited_docstring(Experiments.update)
    def update_experiment(
        self,
        experiment_id: str | None = None,
        changes: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client.experiments.update(experiment_id, changes, **kwargs)

    @inherited_docstring(Experiments.aupdate)
    async def aupdate_experiment(
        self,
        experiment_id: str,
        changes: dict,
    ) -> dict:
        return await self._client.experiments.aupdate(experiment_id, changes)

    @inherited_docstring(Functions.update)
    def update_function(
        self,
        function_id: str | None,
        changes: dict | None = None,
        update_function: str | Callable | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client._functions.update(
            function_id, changes, update_function, **kwargs
        )

    @inherited_docstring(Functions.aupdate)
    async def aupdate_function(
        self,
        function_id: str,
        changes: dict,
        update_function: str | Callable | None = None,
    ) -> dict:
        return await self._client._functions.aupdate(
            function_id, changes, update_function
        )

    @inherited_docstring(Pipelines.update)
    def update_pipeline(
        self,
        pipeline_id: str | None = None,
        changes: dict | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.update(pipeline_id, changes, rev_id, **kwargs)

    @inherited_docstring(Pipelines.aupdate)
    async def aupdate_pipeline(
        self, pipeline_id: str, changes: dict, rev_id: str | None = None
    ) -> dict:
        return await self._client.pipelines.aupdate(pipeline_id, changes, rev_id)

    @inherited_docstring(Models.load, actual_type_override="model")
    def load(self, artifact_id: str | None = None, **kwargs: Any) -> object:
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        return self._client._models.load(artifact_id)

    @inherited_docstring(Models.aload, actual_type_override="model")
    async def aload(self, artifact_id: str) -> object:
        return await self._client._models.aload(artifact_id)

    def download(
        self,
        artifact_id: str | None = None,
        filename: str = "downloaded_artifact.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download the configuration file for an artifact with the specified ID.

        :param artifact_id: unique ID of the model or function
        :type artifact_id: str
        :param filename: name of the file to which the artifact content will be downloaded
        :type filename: str, optional
        :param rev_id: revision ID
        :type rev_id: str, optional
        :param format: format of the content, applicable for models
        :type format: str, optional

        :return: path to the downloaded artifact content
        :rtype: str

        **Examples**

        .. code-block:: python

            client.repository.download(model_id, 'my_model.tar.gz')
            client.repository.download(model_id, 'my_model.json') # if original model was saved as json, works only for xgboost 1.3

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)

        self._validate_type(artifact_id, "artifact_id", str, True)
        self._validate_type(filename, "filename", str, True)

        is_artifact_type = self._check_artifact_type(str(artifact_id))

        if is_artifact_type["model"]:
            return self._client._models.download(artifact_id, filename, rev_id, format)
        if is_artifact_type["function"]:
            return self._client._functions.download(artifact_id, filename, rev_id)
        if is_artifact_type["ai_service"]:
            return self._client._ai_services.download(artifact_id, filename, rev_id)

        raise WMLClientError(
            f"Unexpected type of artifact to download or Artifact with artifact_id: '{artifact_id}' does not exist."
        )

    async def adownload(
        self,
        artifact_id: str,
        filename: str = "downloaded_artifact.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
    ) -> str:
        """Download the configuration file for an artifact with the specified ID asynchronously.

        :param artifact_id: unique ID of the model or function
        :type artifact_id: str
        :param filename: name of the file to which the artifact content will be downloaded
        :type filename: str, optional
        :param rev_id: revision ID
        :type rev_id: str, optional
        :param format: format of the content, applicable for models
        :type format: str, optional

        :return: path to the downloaded artifact content
        :rtype: str

        **Examples**

        .. code-block:: python

            await client.repository.adownload(model_id, 'my_model.tar.gz')
            await client.repository.adownload(model_id, 'my_model.json') # if original model was saved as json, works only for xgboost 1.3

        """

        self._validate_type(artifact_id, "artifact_id", str, True)
        self._validate_type(filename, "filename", str, True)

        is_artifact_type_of = await self._acheck_artifact_type(str(artifact_id))

        if is_artifact_type_of["model"]:
            return await self._client._models.adownload(
                artifact_id, filename, rev_id, format
            )

        if is_artifact_type_of["function"]:
            return await self._client._functions.adownload(
                artifact_id, filename, rev_id
            )

        if is_artifact_type_of["ai_service"]:
            return await self._client._ai_services.adownload(
                artifact_id, filename, rev_id
            )

        raise WMLClientError(
            f"Unexpected type of artifact to download or Artifact with artifact_id: '{artifact_id}' does not exist."
        )

    def _handle_delete_response(
        self, artifact_id: str, response: _requests.Response | httpx.Response
    ) -> Literal["SUCCESS"]:
        match response.status_code:
            case 200 | 204 as success_status_code:
                return self._handle_response(  # type: ignore
                    success_status_code, "delete assets", response
                )
            case 404:
                raise WMLClientError(
                    f"Artifact with artifact_id: '{artifact_id}' does not exist."
                )
            case _:
                raise WMLClientError(
                    "Deletion error for the given id : ", response.text
                )

    def delete(
        self, artifact_id: str | None = None, force: bool = False, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Delete a model, experiment, pipeline, function, or AI service from the repository.

        :param artifact_id: unique ID of the stored model, experiment, function, pipeline, or AI service
        :type artifact_id: str

        :param force: if True, the delete operation will proceed even when the artifact deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            client.repository.delete(artifact_id)

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        Repository._validate_type(artifact_id, "artifact_id", str, True)

        if not force and self._if_deployment_exist_for_asset(artifact_id):
            raise WMLClientError(
                "Cannot delete artifact that has existing deployments. "
                "Please delete all associated deployments and try again"
            )

        params = self._client._params()
        params["purge_on_delete"] = "true"

        response = requests.delete(
            self._client._href_definitions.get_asset_href(artifact_id),
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_delete_response(artifact_id, response)

    async def adelete(
        self, artifact_id: str, force: bool = False
    ) -> Literal["SUCCESS"]:
        """Delete a model, experiment, pipeline, function, or AI service from the repository asynchronously.

        :param artifact_id: unique ID of the stored model, experiment, function, pipeline, or AI service
        :type artifact_id: str

        :param force: if True, the delete operation will proceed even when the artifact deployment exists, defaults to False
        :type force: bool, optional

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]
        :raises: ApiRequestFailure if deletion failed

        **Example:**

        .. code-block:: python

            await client.repository.adelete(artifact_id)

        """

        Repository._validate_type(artifact_id, "artifact_id", str, True)

        if not force and await self._aif_deployment_exist_for_asset(artifact_id):
            raise WMLClientError(
                "Cannot delete artifact that has existing deployments. "
                "Please delete all associated deployments and try again"
            )

        params = self._client._params()
        params["purge_on_delete"] = "true"

        response = await self._client.async_httpx_client.delete(
            self._client._href_definitions.get_asset_href(artifact_id),
            params=params,
            headers=await self._client._aget_headers(),
        )

        return self._handle_delete_response(artifact_id, response)

    def get_details(
        self,
        artifact_id: str | None = None,
        spec_state: SpecStates | None = None,
        artifact_name: str | None = None,
        **kwargs: Any,
    ) -> dict | Generator:
        """Get metadata of stored artifacts. If `artifact_id` and `artifact_name` are not specified,
        the metadata of all models, experiments, functions, pipelines, and ai services is returned.
        If only `artifact_name` is specified, metadata of all artifacts with the name is returned.

        :param artifact_id: unique ID of the stored model, experiment, function, or pipeline
        :type artifact_id: str, optional

        :param spec_state: software specification state, can be used only when `artifact_id` is None
        :type spec_state: SpecStates, optional

        :param artifact_name: name of the stored model, experiment, function, pipeline, or ai service
            can be used only when `artifact_id` is None
        :type artifact_name: str, optional

        :return: metadata of the stored artifact(s)
        :rtype:
            - dict (if artifact_id is not None)
            - {"models": dict, "experiments": dict, "pipeline": dict, "functions": dict, "ai_service": dict} (if artifact_id is None)

        **Examples**

        .. code-block:: python

            details = client.repository.get_details(artifact_id)
            details = client.repository.get_details(artifact_name='Sample_model')
            details = client.repository.get_details()


        Example of getting all repository assets with deprecated software specifications:

        .. code-block:: python

            from ibm_watsonx_ai.lifecycle import SpecStates

            details = client.repository.get_details(spec_state=SpecStates.DEPRECATED)

        """
        artifact_id = _get_id_from_deprecated_uid(
            kwargs, artifact_id, "artifact", can_be_none=True
        )
        Repository._validate_type(artifact_id, "artifact_id", str, False)
        Repository._validate_type(artifact_name, "artifact_name", str, False)

        if artifact_id is not None:
            is_artifact_type_of = self._check_artifact_type(str(artifact_id))

            if is_artifact_type_of["model"]:
                return self.get_model_details(artifact_id)
            if is_artifact_type_of["experiment"]:
                return self.get_experiment_details(artifact_id)
            if is_artifact_type_of["pipeline"]:
                return self.get_pipeline_details(artifact_id)
            if is_artifact_type_of["function"]:
                return self.get_function_details(artifact_id)
            if is_artifact_type_of["ai_service"]:
                return self.get_ai_service_details(artifact_id)

            raise WMLClientError(
                f"Getting artifact details failed. Artifact id: '{artifact_id}' not found."
            )

        model_details = self.get_model_details(
            spec_state=spec_state, model_name=artifact_name
        )

        experiment_details = (
            self.get_experiment_details(experiment_name=artifact_name)
            if not spec_state
            else {"resources": []}
        )

        pipeline_details = (
            self.get_pipeline_details(pipeline_name=artifact_name)
            if not spec_state
            else {"resources": []}
        )

        function_details = self.get_function_details(
            spec_state=spec_state, function_name=artifact_name
        )

        details = {
            "models": model_details,
            "experiments": experiment_details,
            "pipeline": pipeline_details,
            "functions": function_details,
        }

        try:
            details["ai_service"] = self.get_ai_service_details(
                spec_state=spec_state, ai_service_name=artifact_name
            )
        except WMLClientError:
            pass

        return details

    async def aget_details(
        self,
        artifact_id: str | None = None,
        spec_state: SpecStates | None = None,
        artifact_name: str | None = None,
    ) -> dict | AsyncGenerator:
        """Get metadata of stored artifacts asynchronously. If `artifact_id` and `artifact_name` are not specified,
        the metadata of all models, experiments, functions, pipelines, and ai services is returned.
        If only `artifact_name` is specified, metadata of all artifacts with the name is returned.

        :param artifact_id: unique ID of the stored model, experiment, function, or pipeline
        :type artifact_id: str, optional

        :param spec_state: software specification state, can be used only when `artifact_id` is None
        :type spec_state: SpecStates, optional

        :param artifact_name: name of the stored model, experiment, function, pipeline, or ai service
            can be used only when `artifact_id` is None
        :type artifact_name: str, optional

        :return: metadata of the stored artifact(s)
        :rtype:
            - dict (if artifact_id is not None)
            - {"models": dict, "experiments": dict, "pipeline": dict, "functions": dict, "ai_service": dict} (if artifact_id is None)

        **Examples**

        .. code-block:: python

            details = await client.repository.aget_details(artifact_id)
            details = await client.repository.aget_details(artifact_name='Sample_model')
            details = await client.repository.aget_details()


        Example of getting all repository assets with deprecated software specifications:

        .. code-block:: python

            from ibm_watsonx_ai.lifecycle import SpecStates

            details = await client.repository.aget_details(spec_state=SpecStates.DEPRECATED)

        """
        Repository._validate_type(artifact_id, "artifact_id", str, False)
        Repository._validate_type(artifact_name, "artifact_name", str, False)

        if artifact_id is not None:
            is_artifact_type_of = await self._acheck_artifact_type(str(artifact_id))

            if is_artifact_type_of["model"]:
                return await self.aget_model_details(artifact_id)
            if is_artifact_type_of["experiment"]:
                return await self.aget_experiment_details(artifact_id)
            if is_artifact_type_of["pipeline"]:
                return await self.aget_pipeline_details(artifact_id)
            if is_artifact_type_of["function"]:
                return await self.aget_function_details(artifact_id)
            if is_artifact_type_of["ai_service"]:
                return await self.aget_ai_service_details(artifact_id)

            raise WMLClientError(
                f"Getting artifact details failed. Artifact id: '{artifact_id}' not found."
            )

        model_details = await self.aget_model_details(
            spec_state=spec_state, model_name=artifact_name
        )

        experiment_details = (
            await self.aget_experiment_details(experiment_name=artifact_name)
            if not spec_state
            else {"resources": []}
        )

        pipeline_details = (
            await self.aget_pipeline_details(pipeline_name=artifact_name)
            if not spec_state
            else {"resources": []}
        )

        function_details = await self.aget_function_details(
            spec_state=spec_state, function_name=artifact_name
        )

        details = {
            "models": model_details,
            "experiments": experiment_details,
            "pipeline": pipeline_details,
            "functions": function_details,
        }

        try:
            details["ai_service"] = await self.aget_ai_service_details(
                spec_state=spec_state, ai_service_name=artifact_name
            )
        except WMLClientError:
            pass

        return details

    def _get_single_artifact_by_name_from_details(
        self, details: dict, artifact_name: str
    ) -> dict:
        # Check whether 0, 1, or more artifacts were found in 'details' results
        details_by_name = {}
        for artifact_details in details.values():
            if len(artifact_details["resources"]) == 1 and not details_by_name:
                # Found first artifact
                details_by_name = artifact_details["resources"][0]
            elif len(artifact_details["resources"]) > 0 and details_by_name:
                # Found another artifact of different type
                raise WMLClientError(
                    Messages.get_message(
                        artifact_name,
                        message_id="multiple_artifacts_found_by_name",
                    )
                )
            elif len(artifact_details["resources"]) > 1:
                # Found more than 1 artifact of a specific type
                raise WMLClientError(
                    Messages.get_message(
                        artifact_name,
                        message_id="multiple_artifacts_found_by_name",
                    )
                )

        if not details_by_name:
            raise WMLClientError(
                f"Artifact with artifact_name: '{artifact_name}' does not exist."
            )

        return details_by_name

    def get_id_by_name(self, artifact_name: str) -> str:
        """Get the ID of a stored artifact by name.

        :param artifact_name: name of the stored artifact
        :type artifact_name: str

        :return: ID of the stored artifact if exactly one with the 'artifact_name' exists. Otherwise, raise an error.
        :rtype: str

        **Example:**

        .. code-block:: python

            artifact_id = client.repository.get_id_by_name(artifact_name)

        """

        details = cast(dict, self.get_details(artifact_name=artifact_name))
        details_by_name = self._get_single_artifact_by_name_from_details(
            details, artifact_name
        )

        return details_by_name["metadata"]["id"]

    async def aget_id_by_name(self, artifact_name: str) -> str:
        """Get the ID of a stored artifact by name asynchronously.

        :param artifact_name: name of the stored artifact
        :type artifact_name: str

        :return: ID of the stored artifact if exactly one with the 'artifact_name' exists. Otherwise, raise an error.
        :rtype: str

        **Example:**

        .. code-block:: python

            artifact_id = await client.repository.aget_id_by_name(artifact_name)

        """

        details = cast(dict, await self.aget_details(artifact_name=artifact_name))
        details_by_name = self._get_single_artifact_by_name_from_details(
            details, artifact_name
        )

        return details_by_name["metadata"]["id"]

    @inherited_docstring(Models.get_details)
    def get_model_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(
            kwargs, model_id, "model", can_be_none=True
        )
        return self._client._models.get_details(
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            model_name=model_name,
        )

    @inherited_docstring(Models.aget_details)
    async def aget_model_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
    ) -> dict | AsyncGenerator:
        return await self._client._models.aget_details(
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            model_name=model_name,
        )

    @inherited_docstring(Models.get_revision_details)
    def get_model_revision_details(
        self, model_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")
        return self._client._models.get_revision_details(model_id, rev_id)

    @inherited_docstring(Models.aget_revision_details)
    async def aget_model_revision_details(self, model_id: str, rev_id: str) -> dict:
        return await self._client._models.aget_revision_details(model_id, rev_id)

    @inherited_docstring(Experiments.get_details)
    def get_experiment_details(
        self,
        experiment_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        experiment_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client.experiments.get_details(
            experiment_id=experiment_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            experiment_name=experiment_name,
            **kwargs,
        )

    @inherited_docstring(Experiments.aget_details)
    async def aget_experiment_details(
        self,
        experiment_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        experiment_name: str | None = None,
    ) -> dict:
        return await self._client.experiments.aget_details(
            experiment_id=experiment_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            experiment_name=experiment_name,
        )

    @inherited_docstring(Experiments.get_revision_details)
    def get_experiment_revision_details(
        self, experiment_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client.experiments.get_revision_details(
            experiment_id, rev_id, **kwargs
        )

    @inherited_docstring(Experiments.aget_revision_details)
    async def aget_experiment_revision_details(
        self, experiment_id: str, rev_id: str
    ) -> dict:
        return await self._client.experiments.aget_revision_details(
            experiment_id, rev_id
        )

    @inherited_docstring(Functions.get_details)
    def get_function_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client._functions.get_details(
            function_id=function_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            function_name=function_name,
            **kwargs,
        )

    @inherited_docstring(Functions.aget_details)
    async def aget_function_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
    ) -> dict:
        return await self._client._functions.aget_details(
            function_id=function_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            function_name=function_name,
        )

    @inherited_docstring(Functions.get_revision_details)
    def get_function_revision_details(
        self, function_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client._functions.get_revision_details(
            function_id, rev_id, **kwargs
        )

    @inherited_docstring(Functions.aget_revision_details)
    async def aget_function_revision_details(
        self, function_id: str, rev_id: str
    ) -> dict:
        return await self._client._functions.aget_revision_details(function_id, rev_id)

    @inherited_docstring(Pipelines.get_details)
    def get_pipeline_details(
        self,
        pipeline_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        pipeline_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(
            kwargs, pipeline_id, "pipeline", can_be_none=True
        )
        Repository._validate_type(pipeline_id, "pipeline_id", str, False)
        Repository._validate_type(limit, "limit", int, False)
        Repository._validate_type(asynchronous, "asynchronous", bool, False)
        Repository._validate_type(get_all, "get_all", bool, False)
        return self._client.pipelines.get_details(
            pipeline_id=pipeline_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            pipeline_name=pipeline_name,
            **kwargs,
        )

    @inherited_docstring(Pipelines.aget_details)
    async def aget_pipeline_details(
        self,
        pipeline_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        pipeline_name: str | None = None,
    ) -> dict:
        Repository._validate_type(pipeline_id, "pipeline_id", str, False)
        Repository._validate_type(limit, "limit", int, False)
        Repository._validate_type(asynchronous, "asynchronous", bool, False)
        Repository._validate_type(get_all, "get_all", bool, False)
        return await self._client.pipelines.aget_details(
            pipeline_id=pipeline_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            pipeline_name=pipeline_name,
        )

    @inherited_docstring(Pipelines.get_revision_details)
    def get_pipeline_revision_details(
        self, pipeline_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.get_revision_details(
            pipeline_id, rev_id, **kwargs
        )

    @inherited_docstring(Pipelines.aget_revision_details)
    async def aget_pipeline_revision_details(
        self, pipeline_id: str, rev_id: str
    ) -> dict:
        return await self._client.pipelines.aget_revision_details(pipeline_id, rev_id)

    @staticmethod
    @inherited_docstring(Models.get_href)
    def get_model_href(model_details: dict) -> str:
        return Models.get_href(model_details)

    @staticmethod
    @inherited_docstring(Models.get_id)
    def get_model_id(model_details: dict) -> str:
        return Models.get_id(model_details)

    @staticmethod
    @inherited_docstring(
        Experiments.get_id,
        {"experiments.get_details": "repository.get_experiment_details"},
    )
    def get_experiment_id(experiment_details: dict) -> str:
        return Experiments.get_id(experiment_details)

    @staticmethod
    @inherited_docstring(
        Experiments.get_href,
        {"experiments.get_details": "repository.get_experiment_details"},
    )
    def get_experiment_href(experiment_details: dict) -> str:
        return Experiments.get_href(experiment_details)

    @staticmethod
    @inherited_docstring(Functions.get_id)
    def get_function_id(function_details: dict) -> str:
        return Functions.get_id(function_details)

    @staticmethod
    @inherited_docstring(Functions.get_href)
    def get_function_href(function_details: dict) -> str:
        return Functions.get_href(function_details)

    @staticmethod
    @inherited_docstring(
        Pipelines.get_href, {"pipelines.get_details": "repository.get_pipeline_details"}
    )
    def get_pipeline_href(pipeline_details: dict) -> str:
        return Pipelines.get_href(pipeline_details)

    @staticmethod
    @inherited_docstring(Pipelines.get_id)
    def get_pipeline_id(pipeline_details: dict) -> str:
        return Pipelines.get_id(pipeline_details)

    def list(self, framework_filter: str | None = None) -> pandas.DataFrame:
        """Get and list stored models, pipelines, functions, experiments, and AI services in a table/DataFrame format.

        :param framework_filter: get only the frameworks with the desired names
        :type framework_filter: str, optional

        :return: DataFrame with listed names and IDs of stored models
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.repository.list()
            client.repository.list(framework_filter='prompt_tune')

        """

        endpoints = {
            "model": self._client._href_definitions.get_published_models_href(),
            "experiment": self._client._href_definitions.get_experiments_href(),
            "pipeline": self._client._href_definitions.get_pipelines_href(),
            "function": self._client._href_definitions.get_functions_href(),
            "ai_service": self._client._href_definitions.get_ai_services_href(),
        }

        artifact_get = {
            artifact: requests.get(
                url,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            for artifact, url in endpoints.items()
        }

        resources: dict[str, list] = {artifact: [] for artifact in endpoints}

        for artifact in endpoints:
            try:
                response = artifact_get[artifact]
                response_text = self._handle_response(
                    200, f"getting all {artifact}s", response
                )
                resources[artifact] = response_text["resources"]
            except Exception as e:
                self._logger.exception("Error getting %s", artifact, exc_info=e)

        values = []
        for t in endpoints.keys():
            values += [
                (
                    m["metadata"]["id"],
                    m["metadata"]["name"],
                    m["metadata"]["created_at"],
                    m["entity"]["type"] if t == "model" else "-",
                    (
                        t
                        if t != "function" or t != "ai_service"
                        else m["entity"]["type"] + " function"
                    ),
                    self._client.software_specifications._get_state(m),
                    self._client.software_specifications._get_replacement(m),
                )
                for m in resources[t]
            ]

        columns = [
            "ID",
            "NAME",
            "CREATED",
            "FRAMEWORK",
            "TYPE",
            "SPEC_STATE",
            "SPEC_REPLACEMENT",
        ]
        from pandas import DataFrame

        table = DataFrame(data=values, columns=columns)

        table = table.sort_values(by=["CREATED"], ascending=False).reset_index(
            drop=True
        )

        if framework_filter:
            table = table[table["FRAMEWORK"].str.contains(framework_filter)]

        return table

    @inherited_docstring(Models.list)
    def list_models(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> pandas.DataFrame | Generator:
        return self._client._models.list(
            limit=limit, asynchronous=asynchronous, get_all=get_all
        )

    @inherited_docstring(Experiments.list)
    def list_experiments(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client.experiments.list(limit=limit)

    @inherited_docstring(Functions.list)
    def list_functions(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client._functions.list(limit=limit)

    @inherited_docstring(Pipelines.list)
    def list_pipelines(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client.pipelines.list(limit=limit)

    def create_revision(self, artifact_id: str | None = None, **kwargs: Any) -> dict:
        """Create a revision for passed `artifact_id`.

        :param artifact_id: unique ID of a stored model, experiment, function, or pipelines
        :type artifact_id: str

        :return: artifact new revision metadata
        :rtype: dict

        .. deprecated:: 1.3.39
            Use methods corresponding to the artifact type, for example ``create_model_revision()``.

        **Example:**

        .. code-block:: python

            details = client.repository.create_revision(artifact_id)

        """
        create_revision_deprecated_warning = (
            "The create_revision() method is deprecated. "
            "Instead, please use the method corresponding to the artifact type, "
            "for example create_model_revision()."
        )
        warn(create_revision_deprecated_warning, DeprecationWarning)

        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")

        Repository._validate_type(artifact_id, "artifact_id", str, True)

        is_artifact_type_of = self._check_artifact_type(artifact_id)

        if is_artifact_type_of["experiment"]:
            return self.create_experiment_revision(artifact_id)
        if is_artifact_type_of["model"]:
            return self.create_model_revision(artifact_id)
        if is_artifact_type_of["pipeline"]:
            return self.create_pipeline_revision(artifact_id)
        if is_artifact_type_of["function"]:
            return self.create_function_revision(artifact_id)
        if is_artifact_type_of["ai_service"]:
            return self.create_ai_service_revision(artifact_id)

        raise WMLClientError(
            f"Getting artifact details failed. Artifact id: '{artifact_id}' not found."
        )

    @inherited_docstring(Models.list_revisions)
    def list_models_revisions(
        self, model_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.list_revisions(model_id, limit=limit, **kwargs)

    @inherited_docstring(Pipelines.list_revisions)
    def list_pipelines_revisions(
        self, pipeline_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.list_revisions(pipeline_id, limit=limit)

    @inherited_docstring(Functions.list_revisions)
    def list_functions_revisions(
        self, function_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        return self._client._functions.list_revisions(
            function_id, limit=limit, **kwargs
        )

    @inherited_docstring(Experiments.list_revisions)
    def list_experiments_revisions(
        self, experiment_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        return self._client.experiments.list_revisions(
            experiment_id, limit=limit, **kwargs
        )

    @inherited_docstring(Models.promote)
    def promote_model(
        self, model_id: str, source_project_id: str, target_space_id: str
    ) -> str:  # deprecated
        return self._client._models.promote(
            model_id, source_project_id, target_space_id
        )  # type: ignore

    @inherited_docstring(AIServices.store)
    def store_ai_service(
        self, ai_service: str | Callable, meta_props: dict[str, Any]
    ) -> dict:
        return self._client._ai_services.store(ai_service, meta_props)

    @inherited_docstring(AIServices.astore)
    async def astore_ai_service(
        self, ai_service: str | Callable, meta_props: dict[str, Any]
    ) -> dict:
        return await self._client._ai_services.astore(ai_service, meta_props)

    @inherited_docstring(AIServices.get_details)
    def get_ai_service_details(
        self,
        ai_service_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        ai_service_name: str | None = None,
        **kwargs: Any,
    ) -> dict | Generator:
        return self._client._ai_services.get_details(
            ai_service_id=ai_service_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            ai_service_name=ai_service_name,
        )

    @inherited_docstring(AIServices.aget_details)
    async def aget_ai_service_details(
        self,
        ai_service_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        ai_service_name: str | None = None,
    ) -> dict | AsyncGenerator:
        return await self._client._ai_services.aget_details(
            ai_service_id=ai_service_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            ai_service_name=ai_service_name,
        )

    @inherited_docstring(AIServices.update)
    def update_ai_service(
        self,
        ai_service_id: str,
        changes: dict,
        update_ai_service: str | Callable | None = None,
    ) -> dict:
        return self._client._ai_services.update(
            ai_service_id, changes, update_ai_service
        )

    @inherited_docstring(AIServices.aupdate)
    async def aupdate_ai_service(
        self,
        ai_service_id: str,
        changes: dict,
        update_ai_service: str | Callable | None = None,
    ) -> dict:
        return await self._client._ai_services.aupdate(
            ai_service_id, changes, update_ai_service
        )

    @staticmethod
    @inherited_docstring(AIServices.get_id)
    def get_ai_service_id(ai_service_details: dict) -> str:
        return AIServices.get_id(ai_service_details)

    @inherited_docstring(AIServices.list)
    def list_ai_services(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client._ai_services.list(limit=limit)

    @inherited_docstring(AIServices.create_revision)
    def create_ai_service_revision(self, ai_service_id: str, **kwargs: Any) -> dict:
        return self._client._ai_services.create_revision(
            ai_service_id=ai_service_id, **kwargs
        )

    @inherited_docstring(AIServices.acreate_revision)
    async def acreate_ai_service_revision(self, ai_service_id: str) -> dict:
        return await self._client._ai_services.acreate_revision(
            ai_service_id=ai_service_id
        )

    @inherited_docstring(AIServices.get_revision_details)
    def get_ai_service_revision_details(
        self, ai_service_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client._ai_services.get_revision_details(
            ai_service_id, rev_id, **kwargs
        )

    @inherited_docstring(AIServices.aget_revision_details)
    async def aget_ai_service_revision_details(
        self, ai_service_id: str, rev_id: str
    ) -> dict:
        return await self._client._ai_services.aget_revision_details(
            ai_service_id, rev_id
        )

    @inherited_docstring(AIServices.list_revisions)
    def list_ai_service_revisions(
        self, ai_service_id: str, limit: int | None = None
    ) -> pandas.DataFrame:
        return self._client._ai_services.list_revisions(ai_service_id, limit=limit)
