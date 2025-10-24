#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from warnings import warn

from pandas import DataFrame

from ibm_watsonx_ai.foundation_models.schema import BaseSchema
from ibm_watsonx_ai.helpers.connections import (
    ContainerLocation,
    DataConnection,
    FSLocation,
    S3Location,
)
from ibm_watsonx_ai.metanames import RAGOptimizerConfigurationMetaNames
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai.experiment.autoai.engines import RAGEngine
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern import RAGPattern
    from ibm_watsonx_ai.foundation_models.schema import (
        AutoAIRAGCustomModelConfig,
        AutoAIRAGGenerationConfig,
        AutoAIRAGModelConfig,
        AutoAIRAGRetrievalConfig,
    )

__all__ = ["RAGOptimizer"]


class RAGOptimizer:
    """RAGOptimizer class for RAG pattern operation.

    :param name: name for the RAGOptimizer
    :type name: str

    :param engine: engine for remote work on Service instance
    :type engine: RAGEngine

    :param description: description for the RAGOptimizer
    :type description: str, optional

    :param embedding_models: The embedding models to try.
    :type embedding_models: list[str], optional

    :param retrieval_methods: Retrieval methods to be used.
    :type retrieval_methods: list[str], optional

    :param foundation_models: List of foundation models to try. Custom foundation models and model config are also supported for Cloud and CPD >= 5.2.
    :type foundation_models: list[str | dict | AutoAIRAGModelConfig | AutoAIRAGCustomModelConfig], optional

    :param max_number_of_rag_patterns: The maximum number of RAG patterns to create.
    :type max_number_of_rag_patterns: int, optional

    :param optimization_metrics: The metric name(s) to be used for optimization.
    :type optimization_metrics: list[str], optional

    :param generation: Properties describing the generation step.
    :type generation: dict[str, Any] | AutoAIRAGGenerationConfig, optional

    :param retrieval: Retrieval settings to be used.
    :type retrieval: list[dict[str, Any] | AutoAIRAGRetrievalConfig], optional

    """

    def __init__(
        self,
        name: str,
        engine: "RAGEngine",
        description: str | None = None,
        chunking_methods: list[str] | None = None,
        embedding_models: list[str] | None = None,
        retrieval_methods: list[str] | None = None,
        foundation_models: (
            list[str | dict | AutoAIRAGModelConfig | AutoAIRAGCustomModelConfig] | None
        ) = None,
        max_number_of_rag_patterns: int | None = None,
        optimization_metrics: list[str] | None = None,
        chunking: list[dict] | None = None,
        generation: dict[str, Any] | AutoAIRAGGenerationConfig | None = None,
        retrieval: list[dict[str, Any] | AutoAIRAGRetrievalConfig] | None = None,
        **kwargs: dict[str, Any],
    ):
        self._engine = engine

        if chunking_methods is not None:
            chunking_methods_deprecated_warning = "The parameter `chunking_methods` is deprecated, please use `chunking` instead."
            warn(chunking_methods_deprecated_warning, category=DeprecationWarning)

        WMLResource._validate_type(
            foundation_models, "foundation_models", list, mandatory=False
        )
        WMLResource._validate_type(retrieval, "retrieval", list, mandatory=False)

        self._params: dict[str, Any] = {}

        self._params.update(
            {
                "name": name,
                "description": description,
                "chunking": chunking,
                "embedding_models": embedding_models,
                "max_number_of_rag_patterns": max_number_of_rag_patterns,
                "optimization_metrics": optimization_metrics,
            }
        )

        self._handle_generation_and_foundation_models_params(
            generation=generation, foundation_models=foundation_models
        )

        self._handle_retrieval_and_retrieval_methods_params(
            retrieval=retrieval, retrieval_methods=retrieval_methods
        )

        self._engine.initiate_optimizer_metadata(self._params, **kwargs)
        self._engine._params = self._params

        self.ConfigurationMetaNames = RAGOptimizerConfigurationMetaNames()

    def _get_engine(self) -> RAGEngine:
        """Return Engine for development purposes."""
        return self._engine

    def get_params(self) -> dict:
        """Get configuration parameters of RAGOptimizer.

        :return: RAGOptimizer parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_params()

            # Result:
            # {
            #     'name': 'RAG AutoAi ',
            #     'description': 'Sample description',
            #     'max_number_of_rag_patterns': 5,
            #     'optimization_metrics': ['answer_correctness']
            # }

        """
        params_without_none = {k: v for k, v in self._params.items() if v is not None}
        return params_without_none

    def run(
        self,
        input_data_references: list[DataConnection],
        test_data_references: list[DataConnection] | None = None,
        results_reference: DataConnection | None = None,
        vector_store_references: list[DataConnection] | None = None,
        background_mode: bool = True,
    ) -> dict:
        """Create an AutoAI RAG job that will find the best RAG pattern.

        :param input_data_references: Data storage connection details to inform where training data is stored
        :type input_data_references: list[DataConnection]

        :param test_data_references: A set of test data references
        :type test_data_references: list[DataConnection], optional

        :param results_reference: The training results
        :type results_reference: DataConnection, optional

        :param vector_store_references: A set of vector store references
        :type vector_store_references: list[DataConnection], optional

        :param background_mode: Indicator if run() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: run details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            from ibm_watsonx_ai.utils.autoai.enums import TShirtSize
            from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.run(
                input_data_references=[DataConnection(
                    data_asset_id=training_data_asset_id
                )],
                test_data_references=[DataConnection(
                    data_asset_id=test_data_asset_id
                )],
                vector_store_references=[DataConnection(
                    connection_asset_id=milvus_connection_id
                )],
                results_reference=[DataConnection(
                    location=ContainerLocation(
                        path="."
                    )
                )],
                background_mode=False
            )

        """
        results_reference = self._determine_result_reference(
            results_reference, "default_autoai_rag_out"
        )

        results_reference = cast(DataConnection, results_reference)

        return self._engine.run(
            input_data_references=input_data_references,
            results_reference=results_reference,
            test_data_references=test_data_references,
            vector_store_references=vector_store_references,
            background_mode=background_mode,
        )

    def _determine_result_reference(
        self,
        results_reference: DataConnection | None,
        result_path: str,
    ) -> DataConnection:
        if results_reference is None:
            if self._engine._client.CLOUD_PLATFORM_SPACES:
                results_reference = DataConnection(
                    location=ContainerLocation(path=result_path)
                )
            else:
                location = FSLocation()
                client = self._engine._client
                if self._engine._client.default_project_id is None:
                    location.path = location.path.format(
                        option="spaces", id=client.default_space_id
                    )

                else:
                    location.path = location.path.format(
                        option="projects", id=client.default_project_id
                    )
                results_reference = DataConnection(connection=None, location=location)

        elif getattr(results_reference, "type", False) == "fs":
            client = self._engine._client
            results_reference.location = cast(FSLocation, results_reference.location)
            if self._engine._client.default_project_id is None:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="spaces", id=client.default_space_id
                    )
                )
            else:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="projects", id=client.default_project_id
                    )
                )

        results_reference._update_location_path_with_container_id(self._engine._client)

        if not isinstance(
            results_reference.location,
            (S3Location, FSLocation, ContainerLocation),
        ):
            raise TypeError(
                "Unsupported results location type. Results reference can be stored"
                " only on S3Location or FSLocation or ContainerLocation."
            )

        return results_reference

    def cancel_run(self, hard_delete: bool = False) -> str:
        """Cancels a RAG Optimizer run.

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancellation is successful
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)
            rag_optimizer.run(...)

            rag_optimizer.cancel_run()
            # or
            rag_optimizer.cancel_run(hard_delete=True)

        """

        return self._engine.cancel_run(hard_delete=hard_delete)

    def get_run_status(self) -> str:
        """Check status/state of initialized RAGOptimizer run if ran in background mode.

        :return: run status details
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_run_status()

            # Result:
            # 'completed'

        """
        return self._engine.get_run_status()

    def get_run_details(self) -> dict:
        """Get run details.

        :return: RAGOptimizer run details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_run_details()

        """
        return self._engine.get_run_details()

    def summary(self, scoring: str | list[str] | None = None) -> "DataFrame":
        """Return RAGOptimizer summary details.

        :param scoring: scoring metric which user wants to use to sort patterns by,
            when not provided use optimized one
        :type scoring: str | list, optional

        :return: computed patterns and metrics
        :rtype: pandas.DataFrame

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.summary()
            rag_optimizer.summary(scoring='answer_correctness')
            rag_optimizer.summary(scoring=['answer_correctness', 'context_correctness'])

            # Result:
            #                  mean_answer_correctness  ...  ci_high_faithfulness
            # Pattern_Name	                            ...
            # Pattern5                        0.79165   ...                0.5102
            # Pattern1                        0.72915   ...                0.4839
            # Pattern2                        0.64585   ...                0.8333
            # Pattern4                        0.64585   ...                0.5312

        """
        return self._engine.summary(scoring=scoring)

    def get_pattern(self, pattern_name: str | None = None) -> "RAGPattern":
        """Return RAGPattern from RAGOptimizer training.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :return: RAGPattern class for defining, querying and deploying Retrieval-Augmented Generation (RAG) patterns.
        :rtype: RAGPattern

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            pattern_1 = rag_optimizer.get_pattern()
            pattern_2 = rag_optimizer.get_pattern(pattern_name='Pattern2')

        """
        return self._engine.get_pattern(pattern_name=pattern_name)

    def get_pattern_details(self, pattern_name: str | None = None) -> dict:
        """Fetch specific pattern details, e.g. steps etc.

        :param pattern_name: pattern name e.g. Pattern1, if not specified, best pattern parameters will be fetched
        :type pattern_name: str, optional

        :return: pattern parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_pattern_details()
            rag_optimizer.get_pattern_details(pattern_name='Pattern1')

        """
        return self._engine.get_pattern_details(pattern_name=pattern_name)

    def get_inference_notebook(
        self,
        *,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Download specified inference notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved inference notebook
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            inference_notebook_path_1 = rag_optimizer.get_inference_notebook()
            inference_notebook_path_2 = rag_optimizer.get_inference_notebook(
                pattern_name='Pattern1',
                filename='inference_notebook'
            )

        """
        return self._engine.get_inference_notebook(
            pattern_name=pattern_name, local_path=local_path, filename=filename
        )

    def get_indexing_notebook(
        self,
        *,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Download specified indexing notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved indexing notebook
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            indexing_notebook_path_1 = rag_optimizer.get_indexing_notebook()
            indexing_notebook_path_2 = rag_optimizer.get_indexing_notebook(
                pattern_name='Pattern1',
                filename='indexing_notebook'
            )

        """
        return self._engine.get_indexing_notebook(
            pattern_name=pattern_name, local_path=local_path, filename=filename
        )

    def get_logs(self) -> str:
        """
        Get logs of an AutoAI RAG job

        return: path to saved logs
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_logs()

        """
        return self._engine.get_logs()

    def get_evaluation_results(self, pattern_name: str | None = None) -> str:
        """
        Get evaluation results of an AutoAI RAG job

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        return: path to saved evaluation results
        :rtype: str

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_evaluation_results()
            # or
            rag_optimizer.get_evaluation_results(pattern_name='Pattern1')

        """
        return self._engine.get_evaluation_results(pattern_name=pattern_name)

    def _handle_generation_and_foundation_models_params(
        self,
        generation: dict[str, Any] | AutoAIRAGGenerationConfig | None = None,
        foundation_models: (
            list[str | dict | AutoAIRAGModelConfig | AutoAIRAGCustomModelConfig] | None
        ) = None,
    ) -> None:
        if self._engine._client.CPD_version <= 5.1:
            if foundation_models is not None:
                if generation is not None:
                    warning_message = (
                        "Both `foundation_models` and `generation` were provided; using `foundation_models` and "
                        "ignoring `generation`."
                    )
                    warn(warning_message, category=UserWarning)
                if not all(isinstance(fm, str) for fm in foundation_models):
                    raise WMLClientError(
                        "Parameter `foundation_models` must be a list of string for CPD 5.1 or below."
                    )
                self._params["foundation_models"] = foundation_models
            elif generation is not None:
                warning_message = (
                    "In CPD versions ≤ 5.1, the `generation` parameter is not supported. Please update your "
                    "configuration to use the `foundation_models` parameter instead. The functionality of the "
                    "`generation` parameter has been consolidated into `foundation_models` to ensure forward "
                    "compatibility. To suppress this warning, replace all instances of `generation` with "
                    "`foundation_models`."
                )
                warn(warning_message, category=UserWarning)
                if isinstance(generation, BaseSchema):
                    generation = generation.to_dict()

                fm = generation.get("foundation_models", [])

                fm = [el.to_dict() if isinstance(el, BaseSchema) else el for el in fm]

                models = [
                    el["model_id"]
                    for el in fm
                    if isinstance(el, dict) and el.get("model_id")
                ]

                if models:
                    self._params["foundation_models"] = models

            return

        # CPD >= 5.2 and CLOUD scenario
        generation = self._normalize_generation(generation)

        if foundation_models is not None and generation is not None:
            if "foundation_models" in generation:
                warning_message = (
                    "Both `generation` and `foundation_models` were provided. `generation` will take "
                    "precedence.\n\n"
                    "Tip: You can view sample generation parameters using:\n"
                    "    from ibm_watsonx_ai.foundation_models.schema import AutoAIRAGGenerationConfig\n"
                    "    AutoAIRAGGenerationConfig.get_sample_params()\n"
                )
                warn(warning_message, category=UserWarning)
                generation = {
                    **generation,
                    "foundation_models": self._normalize_fm_list(
                        generation["foundation_models"]
                    ),
                }
            else:
                warning_message = (
                    "`foundation_models` was provided separately and has been merged into `generation` because "
                    "`generation` did not include a `foundation_models` key. To silence this warning, pass "
                    "foundation models inside `generation`."
                )
                warn(warning_message, category=UserWarning)
                generation = {
                    **generation,
                    "foundation_models": self._normalize_fm_list(foundation_models),
                }

        if generation is None and foundation_models is not None:
            warning_message = (
                "`foundation_models` is deprecated and will be removed in a future release. "
                "Use `generation` instead.\n\n"
                "Tip: You can view sample generation parameters using:\n"
                "    from ibm_watsonx_ai.foundation_models.schema import AutoAIRAGGenerationConfig\n"
                "    AutoAIRAGGenerationConfig.get_sample_params()\n"
            )
            warn(warning_message, category=DeprecationWarning)
            generation = {
                "foundation_models": self._normalize_fm_list(foundation_models)
            }

        if generation:
            self._params["generation"] = generation

    def _handle_retrieval_and_retrieval_methods_params(
        self,
        retrieval: list[dict[str, Any] | AutoAIRAGRetrievalConfig] | None = None,
        retrieval_methods: list[str] | None = None,
    ) -> None:
        if self._engine._client.CPD_version <= 5.1:
            if retrieval_methods is not None:
                if retrieval is not None:
                    warning_message = (
                        "Both `retrieval` and `retrieval_methods` were provided; using `retrieval_methods` "
                        "and ignoring `retrieval`."
                    )
                    warn(warning_message, category=UserWarning)
                self._params["retrieval_methods"] = retrieval_methods
            elif retrieval is not None:
                warning_message = (
                    "In CPD versions ≤ 5.1, the `retrieval` parameter is not supported. Please update your "
                    "configuration to use the `retrieval_methods` parameter instead. The functionality of the "
                    "`retrieval` parameter has been consolidated into `retrieval_methods` to ensure forward "
                    "compatibility. To suppress this warning, replace all instances of `retrieval` with "
                    "`retrieval_methods`."
                )
                warn(warning_message, category=UserWarning)
                retrieval_dict = [
                    el.to_dict() if isinstance(el, BaseSchema) else el
                    for el in retrieval
                ]
                retrieval = [
                    el["method"]
                    for el in retrieval_dict
                    if isinstance(el, dict) and el.get("method")
                ]

                self._params["retrieval_methods"] = retrieval

            return

        # CPD >= 5.2 and CLOUD scenario
        if retrieval_methods is not None:
            if retrieval is not None:
                warning_message = (
                    "Both `retrieval` and `retrieval_methods` were provided; using `retrieval` "
                    "and ignoring `retrieval_methods`."
                )
                warn(warning_message, category=UserWarning)
            else:
                warning_message = (
                    "The `retrieval_methods` parameter is deprecated and will be removed in a future release. "
                    "Please pass retrieval configuration via the `retrieval` parameter instead."
                )
                warn(warning_message, category=DeprecationWarning)
                retrieval = [{"method": m} for m in retrieval_methods]

        if retrieval is not None:
            self._params["retrieval"] = self._normalize_retrieval(retrieval)

    def _normalize_generation(
        self, generation: dict | BaseSchema | None
    ) -> dict[str, Any] | None:
        if generation is None:
            return None

        if isinstance(generation, BaseSchema):
            gen = generation.to_dict()
        elif isinstance(generation, dict):
            gen = dict(generation)
        else:
            raise WMLClientError(
                f"Unsupported type for `generation`: {type(generation)}, expected {dict}"
            )

        if "foundation_models" in gen:
            WMLResource._validate_type(
                gen["foundation_models"],
                "generation['foundation_models']",
                list,
                mandatory=False,
            )
            gen["foundation_models"] = self._normalize_fm_list(gen["foundation_models"])

        return gen

    @staticmethod
    def _normalize_fm_list(fms: list) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fm in fms:
            match fm:
                case BaseSchema():
                    out.append(fm.to_dict())
                case dict():
                    out.append(dict(fm))
                case str():
                    out.append({"model_id": fm})
                case _:
                    raise WMLClientError(
                        f"Invalid item type '{type(fm)}' provided in `foundation_models` list, expected {str}, {dict} or {BaseSchema} type class."
                    )

        return out

    @staticmethod
    def _normalize_retrieval(retrieval: list) -> list[dict[str, Any]]:
        norm: list[dict[str, Any]] = []
        for r in retrieval:
            match r:
                case BaseSchema():
                    norm.append(r.to_dict())  # type: ignore[union-attr]
                case dict():
                    norm.append(dict(r))
                case _:
                    raise WMLClientError(
                        f"Invalid item type '{type(r)}' provided in `retrieval` list, expected {dict} or {BaseSchema} type class."
                    )

        return norm
