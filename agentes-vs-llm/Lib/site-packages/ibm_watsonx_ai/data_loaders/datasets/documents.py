#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = [
    "DocumentsIterableDataset",
]

import logging
import sys
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from ibm_watsonx_ai.data_loaders.datasets.base_documents import (
    DEFAULT_DOCUMENTS_SAMPLING_TYPE,
    DEFAULT_SAMPLE_SIZE_LIMIT,
    BaseDocumentsIterableDataset,
)
from ibm_watsonx_ai.data_loaders.text_loader import (
    TextLoader,
)
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from ibm_watsonx_ai.helpers.connections import DataConnection

logger = logging.getLogger(__name__)


class DocumentsIterableDataset(BaseDocumentsIterableDataset):
    """
    This dataset is an Iterable stream of documents using an underneath Flight Service.
    It can download documents asynchronously and serve them to you from a generator.

    Supported types of documents:
        - **text/plain** (".txt" file extension) - plain structured text
        - **docx** (".docx" file extension) - standard Word style file
        - **pdf** (".pdf" file extension) - standard pdf document
        - **html** (".html" file extension) - saved html side
        - **markdown** (".md" file extension) - plain text formatted with markdown
        - **pptx** (".pptx" file extension) - standard PowerPoint style file
        - **json** (".json" file extension) - standard json file
        - **yaml** (".yaml" file extension) - standard yaml file
        - **xml** (".xml" file extension) - standard xml file
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

            iterable_dataset = DocumentsIterableDataset(connections=connections,
                                                        enable_sampling=True,
                                                        sampling_type='random',
                                                        sample_size_limit = 1GB)

    **Example: read all documents/no subsampling**

        .. code-block:: python

            connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

            iterable_dataset = DocumentsIterableDataset(connections=connections,
                                                        enable_sampling=False)

    **Example: context based sampling**

            .. code-block:: python

                connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

                iterable_dataset = DocumentsIterableDataset(connections=connections,
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

    def _load_doc(self, doc: RemoteDocument) -> Document:
        doc.download()
        return TextLoader(doc).load()

    def _get_element_size(self, el: Any) -> int:
        return sys.getsizeof(el.page_content)
