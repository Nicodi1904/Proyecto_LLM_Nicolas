#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from ibm_watsonx_ai.utils.utils import get_from_json

__all__ = [
    "TabularDocumentsIterableDataset",
]

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from ibm_watsonx_ai.data_loaders.datasets.base_documents import (
    DEFAULT_DOCUMENTS_SAMPLING_TYPE,
    DEFAULT_SAMPLE_SIZE_LIMIT,
    BaseDocumentsIterableDataset,
)
from ibm_watsonx_ai.data_loaders.datasets.tabular import TabularIterableDataset
from ibm_watsonx_ai.data_loaders.experiment import ExperimentDataLoader
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument

if TYPE_CHECKING:
    from pandas import DataFrame

    from ibm_watsonx_ai.helpers.connections import DataConnection

logger = logging.getLogger(__name__)


class TabularDocumentsIterableDataset(BaseDocumentsIterableDataset):
    """
    This dataset is an Iterable stream of dataframes using an underneath Flight Service.
    It can download dataframes asynchronously and serve them to you from a generator.

    Supported types of documents:
        - **csv** (".csv" file extension) - standard csv file
        - **excel** (".xlsx" file extension) - standard Excel file

    :param connections: list of connections to the documents
    :type connections: list[DataConnection]

    :param enable_sampling: if set to `True`, will enable sampling, default: True
    :type enable_sampling: bool

    :param include_subfolders: if set to `True`, all documents in subfolders of connections locations will be included, default: False
    :type include_subfolders: bool, optional

    :param sample_size_limit: upper limit for documents to be downloaded in bytes, default: 1 GB
    :type sample_size_limit: int

    :param sampling_type: a sampling strategy on how to read the data,
        check the `DocumentsSamplingTypes` enum class for more options
    :type sampling_type: str

    :param total_size_limit: upper limit for documents to be downloaded in Bytes, default: 1 GB,
        if more than one of: `total_size_limit`, `total_ndocs_limit` are set,
        then data are limited to the lower threshold.
    :type total_size_limit: int

    :param total_ndocs_limit: upper limit for documents to be downloaded in a number of rows,
        if more than one of: `total_size_limit`, `total_nrows_limit` are set,
        then data are limited to the lower threshold.
    :type total_ndocs_limit: int, optional

    :param benchmark_dataset: dataset of benchmarking data with IDs in the `document_ids` column corresponding
        to the names of documents in the `connections` list
    :type benchmark_dataset: pd.DataFrame, optional

    :param error_callback: error callback function, to handle the exceptions from document loading,
        as arguments are passed document_id and exception
    :type error_callback: function (str, Exception) -> None, optional

    :param api_client: initialized APIClient object with set project or space ID. If the DataConnection object in list
     connections does not have a set API client, then the api_client object is used for reading data.
    :type api_client: APIClient, optional


    **Example: default sampling - read up to 1 GB of random documents**

        .. code-block:: python

            connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

            iterable_dataset = TabularDocumentsIterableDataset(connections=connections,
                                                        enable_sampling=True,
                                                        sampling_type='random',
                                                        sample_size_limit = 1GB)

    **Example: read all documents/no subsampling**

        .. code-block:: python

            connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

            iterable_dataset = TabularDocumentsIterableDataset(connections=connections,
                                                        enable_sampling=False)

    **Example: context based sampling**

            .. code-block:: python

                connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

                iterable_dataset = TabularDocumentsIterableDataset(connections=connections,
                                                            enable_sampling=True,
                                                            sampling_type='benchmark_driven',
                                                            sample_size_limit = 1GB,
                                                            benchmark_dataset=pd.DataFrame(
                                                                data={
                                                                    "question": [
                                                                        "What foundation models are available in watsonx.ai ?"
                                                                    ],
                                                                    "correct_answers": [
                                                                        [
                                                                            "The following models are available in watsonx.ai: ..."
                                                                        ]
                                                                    ],
                                                                    "correct_answer_document_ids": ["sample_pdf_file.pdf"],
                                                                }))

    """

    def __init__(
        self,
        *,
        connections: list[DataConnection],
        enable_sampling: bool = True,
        include_subfolders: bool = False,
        sample_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        sampling_type: str = DEFAULT_DOCUMENTS_SAMPLING_TYPE,
        total_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        total_ndocs_limit: int | None = None,
        benchmark_dataset: pd.DataFrame | None = None,
        error_callback: Callable[[str, Exception], None] = None,
        **kwargs: Any,
    ) -> None:
        BaseDocumentsIterableDataset.__init__(
            self,
            connections=connections,
            enable_sampling=enable_sampling,
            include_subfolders=include_subfolders,
            sample_size_limit=sample_size_limit,
            sampling_type=sampling_type,
            total_size_limit=total_size_limit,
            total_ndocs_limit=total_ndocs_limit,
            benchmark_dataset=benchmark_dataset,
            error_callback=error_callback,
            **kwargs,
        )
        self.tabular_iterable_args = kwargs.get("tabular_iterable_args", {})

    def _prepare_file_type_flavored_tabular_iterable_args(self, doc: RemoteDocument):
        tabular_iterable_args = deepcopy(self.tabular_iterable_args)

        if (
            doc.document_id.endswith(".xlsx") or doc.document_id.endswith(".xls")
        ) and "file_format" not in get_from_json(
            tabular_iterable_args, ["flight_parameters", "interaction_properties"], {}
        ):
            if "flight_parameters" not in tabular_iterable_args:
                tabular_iterable_args["flight_parameters"] = {}

            if (
                "interaction_properties"
                not in tabular_iterable_args["flight_parameters"]
            ):
                tabular_iterable_args["flight_parameters"]["interaction_properties"] = {
                    "file_format": "excel"
                }

        return tabular_iterable_args

    def _load_doc(self, doc: RemoteDocument) -> ExperimentDataLoader:
        tabular_iterable_args = self._prepare_file_type_flavored_tabular_iterable_args(
            doc
        )

        return ExperimentDataLoader(
            dataset=TabularIterableDataset(
                connection=doc.connection, **tabular_iterable_args
            )
        )

    def _get_element_size(self, el: "DataFrame") -> int:
        return el.memory_usage(index=True).sum()
