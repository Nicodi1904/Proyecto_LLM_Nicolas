#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generator, Literal, overload
from warnings import warn

from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.utils.utils import StrEnum
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class FoundationModelsManager(WMLResource):
    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)
        self._client = client

    @cached_property
    def TextModels(self):
        return StrEnum("TextModels", self._get_model_dict("text_generation"))

    @cached_property
    def ChatModels(self):
        return StrEnum("ChatModels", self._get_model_dict("text_chat"))

    @cached_property
    def EmbeddingModels(self):
        return StrEnum("EmbeddingModels", self._get_model_dict("embedding"))

    @cached_property
    def PromptTunableModels(self):
        return StrEnum("PromptTunableModels", self._get_model_dict("prompt_tuning"))

    @cached_property
    def RerankModels(self):
        return StrEnum("RerankModels", self._get_model_dict("rerank"))

    @cached_property
    def TimeSeriesModels(self):
        return StrEnum("TimeSeriesModels", self._get_model_dict("time_series_forecast"))

    @cached_property
    def AudioTranscriptionsModels(self):
        return StrEnum(
            "AudioTranscriptionsModels", self._get_model_dict("audio_transcriptions")
        )

    def _get_spec(
        self,
        url: str,
        operation_name: str,
        error_msg_id: str,
        model_id: str | None = None,
        limit: int | None = 50,
        filters: str | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        tech_preview: bool = False,
    ) -> dict | Generator | None:
        params = self._client._params(skip_userfs=True, skip_space_project_chk=True)
        if filters:
            params.update({"filters": filters})

        if tech_preview:
            params.update({"tech_preview": True})

        try:
            if model_id:
                result = self._get_with_or_without_limit(
                    url,
                    limit=None,
                    op_name=operation_name,
                    query_params=params,
                    _all=True,
                    _async=False,
                    skip_space_project_chk=True,
                )

                if isinstance(model_id, Enum):
                    model_id = model_id.value

                model_res = [
                    res for res in result["resources"] if res["model_id"] == model_id
                ]

                if len(model_res) > 0:
                    return model_res[0]
                else:
                    return None
            else:
                return self._get_with_or_without_limit(
                    url=url,
                    limit=limit,
                    op_name=operation_name,
                    query_params=params,
                    _async=asynchronous,
                    _all=get_all,
                    skip_space_project_chk=True,
                )
        except WMLClientError as e:
            raise WMLClientError(
                Messages.get_message(
                    self._client.credentials.url,
                    message_id=error_msg_id,
                ),
                e,
            )

    def get_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        filters: str | None = "function_text_generation,!lifecycle_withdrawn:and",
        **kwargs: Any,
    ) -> dict | Generator | None:
        """
        Retrieves a list of specifications for a deployed foundation model.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str or ModelTypes, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param filters: a set of filters to specify the list of models, default to ``function_text_generation``
            Filters are described by the following pattern:

            ``pattern: tfilter[,tfilter][:(or|and)]``
            ``tfilter: filter | !filter``

            * ``filter`` – requires existence of the filter.
            * ``!filter`` – requires absence of the filter.

            **Available filters:**

            .. list-table::
               :header-rows: 1
               :widths: 20 55 25

               * - Filter
                 - Description
                 - Example
               * - ``modelid_*``
                 - Filters by model id. Selects a model with a specific model id.
                 - ``modelid_ibm/granite-3-3-8b-instruct``
               * - ``provider_*``
                 - Filters by provider. Selects all models with a specific provider.
                 - ``provider_IBM``
               * - ``source_*``
                 - Filters by source. Selects all models with a specific source.
                 - ``source_IBM``
               * - ``input_tier_*``
                 - Filters by input tier. Selects all models with a specific input tier.
                 - ``input_tier_class_1``
               * - ``output_tier_*``
                 - Filters by output tier. Selects all models with a specific output tier.
                 - ``output_tier_class_c1``
               * - ``tier_*``
                 - Filters by tier. Selects all models with a specific input or output tier.
                 - ``tier_class_12``
               * - ``task_*``
                 - Filters by task id. Selects all models supporting a specific task id.
                 - ``task_generation``
               * - ``lifecycle_*``
                 - Filters by lifecycle state. Selects all models in the specified lifecycle state.
                 - ``lifecycle_available``
               * - ``function_*``
                 - Filters by function. Selects all models supporting a specific function.
                 - ``function_text_chat``
        :type filters: str, list[str], optional

        :return: list of specifications for the deployed foundation model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            # GET ALL MODEL SPECS
            client.foundation_models.get_model_specs()

            # GET MODEL SPECS BY MODEL_ID
            client.foundation_models.get_model_specs(model_id="google/flan-ul2")

            # GET MODEL SPECS WITH ONE FILTER
            client.foundation_models.get_model_specs(filters="function_text_generation")

            # GET MODEL SPECS WITH MULTIPLE FILTERS
            client.foundation_models.get_model_specs(filters="function_text_generation,function_text_chat,!lifecycle_withdrawn:and")
        """
        WMLResource._validate_type(filters, "filters", str, mandatory=False)

        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available foundation models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            filters=(None if self._client.CPD_version < 5.0 else filters),
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            tech_preview=kwargs.get("tech_preview", False),
        )

    def get_chat_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to retrieve the list of chat foundation models specifications.

        :param model_id: id of the model, defaults to None (all models specs are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of deployed foundation model specs
        :rtype: dict or generator

        **Example**

        .. code-block:: python

            # GET CHAT MODEL SPECS
            client.foundation_models.get_chat_model_specs()

            # GET CHAT MODEL SPECS BY MODEL_ID
            client.foundation_models.get_chat_model_specs(model_id="ibm/granite-13b-chat-v2")
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available chat models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_text_chat,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_chat_function_calling_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to retrieve the list of chat foundation models specifications with function calling support .

        :param model_id: id of the model, defaults to None (all models specs are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of deployed foundation model specs
        :rtype: dict or generator

        **Example**

        .. code-block:: python

            # GET CHAT FUNCTION CALLING MODEL SPECS
            client.foundation_models.get_chat_function_calling_model_specs()

            # GET CHAT FUNCTION CALLING MODEL SPECS BY MODEL_ID
            client.foundation_models.get_chat_function_calling_model_specs(model_id="meta-llama/llama-3-1-70b-instruct")
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available chat function calling models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="task_function_calling,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_audio_chat_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to retrieve the list of audio chat foundation models specifications.

        :param model_id: id of the model, defaults to None (all models specs are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of deployed foundation model specs
        :rtype: dict or generator

        **Example**

        .. code-block:: python

            # GET AUDIO CHAT MODEL SPECS
            client.foundation_models.get_audio_chat_model_specs()

            # GET AUDIO CHAT MODEL SPECS BY MODEL_ID
            client.foundation_models.get_audio_chat_model_specs(model_id="ibm/granite-speech-3-3-8b")
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available audio chat models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_audio_chat,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_text_generation_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to retrieve the list of text generation foundation models specifications.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str or ModelTypes, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of specifications for the deployed foundation model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            # GET TEXT GENERATION MODEL SPECS
            client.foundation_models.get_text_generation_model_specs()

            # GET TEXT GENERATION MODEL SPECS BY MODEL_ID
            client.foundation_models.get_text_generation_model_specs(model_id="google/flan-ul2")
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available text generation foundation models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            filters=(
                None
                if self._client.CPD_version < 5.0
                else "function_text_generation,!lifecycle_withdrawn:and"
            ),
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_custom_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """Get details on available custom model(s) as a dictionary or as a generator (``asynchronous``).
        If ``asynchronous`` or ``get_all`` is set, then ``model_id`` is ignored.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: details of supported custom models, None if no supported custom models are found for the given model_id
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_custom_models_spec()
            client.foundation_models.get_custom_models_spec()
            client.foundation_models.get_custom_models_spec(model_id='mistralai/Mistral-7B-Instruct-v0.2')
            client.foundation_models.get_custom_models_spec(limit=20)
            client.foundation_models.get_custom_models_spec(limit=20, get_all=True)
            for spec in client.foundation_models.get_custom_model_specs(limit=20, asynchronous=True, get_all=True):
                print(spec, end="")

        """
        if self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(
                Messages.get_message(message_id="custom_models_cloud_scenario")
            )

        model_usage_warning = (
            "Model needs to be first stored via client.repository.store_model(model_id, meta_props=metadata) "
            "and deployed via client.deployments.create(asset_id, metadata) to be used."
        )
        warn(model_usage_warning)

        return self._get_spec(
            url=self._client._href_definitions.get_fm_custom_foundation_models_href(),
            operation_name="Get custom model specs",
            error_msg_id="custom_models_no_model_specs",
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_embeddings_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Retrieves the specifications of an embeddings model.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: specifications of the embeddings model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_embeddings_model_specs()
            client.foundation_models.get_embeddings_model_specs('ibm/slate-125m-english-rtrvr')
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available embedding models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_embedding,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_time_series_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Retrieves the specifications of an time series model.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: specifications of the time series model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_time_series_model_specs()
            client.foundation_models.get_time_series_model_specs('ibm/granite-ttm-1536-96-r2')
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available time series models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_time_series_forecast,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_rerank_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Retrieves the specifications of a rerank model.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: specifications of the rerank model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_rerank_model_specs()
            client.foundation_models.get_rerank_model_specs('ibm/slate-125m-english-rtrvr-v2')

        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available rerank models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_rerank,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_audio_transcriptions_model_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Retrieves the specifications of an audio transcriptions model.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: specifications of the audio transcriptions model
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_audio_transcriptions_model_specs()
            client.foundation_models.get_audio_transcriptions_model_specs('openai/whisper-tiny')

        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available audio transcriptions models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_audio_transcriptions,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_base_foundation_model_deployable_specs(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Retrieve the specifications of a base deployable foundation models

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: specifications of the base foundation model deployable
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_base_foundation_model_deployable_specs()
            client.foundation_models.get_base_foundation_model_deployable_specs('meta-llama/llama-3-1-8b')

        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available base foundation model deployable",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_base_foundation_model_deployable,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    @overload
    def get_model_specs_with_prompt_tuning_support(
        self,
        model_id: str | None = ...,
        limit: int | None = ...,
        asynchronous: Literal[False] = False,
        get_all: bool = ...,
    ) -> dict | None: ...

    @overload
    def get_model_specs_with_prompt_tuning_support(
        self,
        model_id: str | None,
        limit: int | None,
        asynchronous: Literal[True],
        get_all: bool,
    ) -> Generator: ...

    def get_model_specs_with_prompt_tuning_support(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Queries the details of deployed foundation models with prompt tuning support.

        :param model_id: id of the model, defaults to None (all models specifications are returned)
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of specifications of a deployed foundation model with prompt tuning support
        :rtype: dict or generator

        **Example:**

        .. code-block:: python

            client.foundation_models.get_model_specs_with_prompt_tuning_support()
            client.foundation_models.get_model_specs_with_prompt_tuning_support('google/flan-t5-xl')
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available foundation models",
            error_msg_id="fm_prompt_tuning_no_model_specs",
            model_id=model_id,
            filters="function_prompt_tune_trainable,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_model_specs_with_fine_tuning_support(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to query the details of the deployed foundation models with fine-tuning support.

        :param model_id: id of the model, defaults to None (all models specs are returned)
        :type model_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param asynchronous:  if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of deployed foundation model specs with fine-tuning support
        :rtype: dict or generator

        **Example**

        .. code-block:: python

            client.foundation_models.get_model_specs_with_fine_tuning_support()
            client.foundation_models.get_model_specs_with_fine_tuning_support('bigscience/bloom')
        """
        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available foundation models with fine tuning support",
            error_msg_id="fm_fine_tuning_no_model_specs",
            model_id=model_id,
            filters="function_fine_tune_trainable,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_model_specs_with_lora_fine_tuning_support(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict | Generator | None:
        """
        Operations to query the details of the deployed foundation models with lora fine-tuning support.

        :param model_id: id of the model, defaults to None (all models specs are returned)
        :type model_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param asynchronous:  if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: list of deployed foundation model specs with lora fine-tuning support
        :rtype: dict or generator

        **Example**

        .. code-block:: python

            client.foundation_models.get_model_specs_with_lora_fine_tuning_support()
            client.foundation_models.get_model_specs_with_lora_fine_tuning_support('bigscience/bloom')
        """

        return self._get_spec(
            url=self._client._href_definitions.get_fm_specifications_href(),
            operation_name="Get available foundation models with lora fine tuning support",
            error_msg_id="fm_fine_tuning_no_model_specs",
            model_id=model_id,
            filters="function_lora_fine_tune_trainable,!lifecycle_withdrawn:and",
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
        )

    def get_model_lifecycle(self, model_id: str, **kwargs: Any) -> list | None:
        """
        Retrieves a list of lifecycle data of a foundation model.

        :param model_id: id of the model
        :type model_id: str

        :return: list of lifecycle data of a foundation model
        :rtype: list

        **Example:**

        .. code-block:: python

            client.foundation_models.get_model_lifecycle(
                model_id="ibm/granite-13b-instruct-v2"
                )
        """
        model_spec = self.get_model_specs(
            model_id, tech_preview=kwargs.get("tech_preview", False)
        )
        return model_spec.get("lifecycle") if model_spec is not None else None

    def _get_model_dict(
        self,
        model_type: Literal[
            "base",
            "embedding",
            "prompt_tuning",
            "text_chat",
            "rerank",
            "time_series_forecast",
            "audio_transcriptions",
            "text_generation",
        ],
    ) -> dict:
        """
        Retrieves the dictionary of models to Enum.

        :param model_type: type of model function
        :type model_type: Literal["base", "embedding", "prompt_tuning", "text_chat", "rerank", "time_series_forecast", "audio_transcriptions", "text_generation"]

        :return: dictionary of models to Enum
        :rtype: dict
        """
        function_dict = {
            "base": self.get_model_specs,
            "embedding": self.get_embeddings_model_specs,
            "prompt_tuning": self.get_model_specs_with_prompt_tuning_support,
            "text_chat": self.get_chat_model_specs,
            "rerank": self.get_rerank_model_specs,
            "time_series_forecast": self.get_time_series_model_specs,
            "audio_transcriptions": self.get_audio_transcriptions_model_specs,
            "text_generation": self.get_text_generation_model_specs,
        }
        model_specs_dict = {}
        for model_spec in function_dict[model_type]()["resources"]:
            if "model_id" in model_spec:
                model_specs_dict[
                    model_spec["model_id"].split("/")[-1].replace("-", "_").upper()
                ] = model_spec["model_id"]
        return model_specs_dict
