#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import copy
from typing import Any, TypeAlias, cast

from langchain_core.documents import Document

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import (
    MissingExtension,
    VectorStoreSerializationError,
)

try:
    from langchain_core.embeddings import Embeddings as LCEmbeddings
    from langchain_db2 import DB2VS
    from langchain_db2.db2vs import clear_table

except ImportError as exc:
    raise MissingExtension(
        "langchain_db2",
        reason="Please install `ibm-watsonx-ai` with flag `rag`: \n `pip install -U 'ibm-watsonx-ai[rag]'`",
    ) from exc

import logging

from ibm_watsonx_ai import APIClient

logger = logging.getLogger(__name__)

# Type Alias
EmbeddingType: TypeAlias = BaseEmbeddings | LCEmbeddings


class DB2VectorStore(LangChainVectorStoreAdapter):
    """DB2VectorStore vector store client for a RAG pattern.

    :param api_client: api client is required if connecting by connection_id, defaults to None
    :type api_client: APIClient, optional

    :param connection_id: connection asset ID, defaults to None
    :type connection_id: str, optional

    :param vector_store: initialized langchain_db2 vector store, defaults to None
    :type vector_store: langchain_db2.DB2VS, optional

    :param embedding_function: dense embedding function, defaults to None
    :type embedding_function: BaseEmbeddings | LCEmbeddings, optional

    :param table_name: name of the DB2 table name, defaults to None
    :type table_name: str, optional

    :param kwargs: keyword arguments that will be directly passed to `langchain_db2.DB2VS` constructor
    :type kwargs: Any, optional

    **Example:**

    To connect, provide the connection asset ID.
    You can use custom embeddings to add and search documents.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import DB2VectorStore
        from ibm_watsonx_ai.foundation_models.embeddings import Embeddings

        credentials = Credentials(
            api_key = IAM_API_KEY,
            url = "https://us-south.ml.cloud.ibm.com"
        )

        api_client = APIClient(credentials, project_id="<PROJECT_ID>")

        embedding = Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            api_client=api_client
        )

        vector_store = DB2VectorStore(
            api_client,
            connection_id='***',
            collection_name='my_test_collection',
            embedding_function=embedding
        )

        vector_store.add_documents([
            {'content': 'document one content', 'metadata':{'url':'ibm.com'}},
            {'content': 'document two content', 'metadata':{'url':'ibm.com'}}
        ])
        # ['4CDDAF00329B3DF9', 'B8AE97421A8857E7']

        vector_store.search('one', k=1)
        # [Document(metadata={'url': 'ibm.com'}, page_content='document one content')]

    """

    def __init__(
        self,
        api_client: APIClient | None = None,
        *,
        connection_id: str | None = None,
        vector_store: DB2VS | None = None,
        embedding_function: EmbeddingType | None = None,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._connection_id = connection_id
        self._client = api_client

        self._is_serializable = not bool(vector_store)
        self._embedding_function = embedding_function
        self._table_name = table_name
        self._additional_kwargs = kwargs

        # import is not at top-level of file due to circular import
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.vector_store_connector import (  # noqa: E501, PLC0415
            VectorStoreConnector,
            VectorStoreDataSourceType,
        )

        if vector_store is None:
            if self._client is not None and self._connection_id is not None:
                self._datasource_type, connection_properties = self._connect_by_type(
                    cast(str, self._connection_id)
                )
            else:
                self._datasource_type, connection_properties = (
                    VectorStoreDataSourceType.DB2,
                    {},
                )

            logger.info("Initializing vector store of type: %s", self._datasource_type)

            self._properties = {
                **connection_properties,
                **self._additional_kwargs,
                "embedding_function": self._embedding_function,
                "table_name": self._table_name,
            }

            self._properties = VectorStoreConnector(
                self._properties
            )._get_db2_connection_params()

            vector_store = DB2VS(**self._properties)
        else:
            self._datasource_type = (
                VectorStoreConnector.get_type_from_langchain_vector_store(vector_store)
            )

        super().__init__(
            vector_store=vector_store,
        )

    def get_client(self) -> DB2VS:
        """Get langchain_db2.DB2VS instance."""
        return super().get_client()

    def clear(self) -> None:
        """
        Clear table by removing all records.
        """
        db2_client = self.get_client()

        clear_table(client=db2_client.client, table_name=db2_client.table_name)

    def count(self) -> int:
        """
        Count number of records in table.
        """
        ids = self.get_client().get_pks()

        return len(ids) if ids else 0

    def add_documents(
        self, content: list[str] | list[dict] | list[Document], **kwargs: Any
    ) -> list[str]:
        """
        Embed documents and add to the vectorstore.

        :param content: Documents to add to the vectorstore.
        :type content: list[str] | list[dict] | list[langchain_core.documents.Document]

        :return: List of IDs of the added texts.
        :rtype: list[str]
        """
        ids, docs = self._process_documents(content)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        if len(texts) == 0:
            return []

        db2 = self.get_client()

        return db2.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def to_dict(self) -> dict:
        """Serialize ``DB2VectorStore`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict

        :raises VectorStoreSerializationError: when instance is not serializable
        """

        if not self._is_serializable:
            raise VectorStoreSerializationError(
                "Serialization is not available when passing vector store instance in `DB2VectorStore` constructor."
            )
        embedding = self._embedding_function
        if embedding is None:
            embedding_f = None
        else:
            watsonx = getattr(embedding, "watsonx_embed", None)
            to_dict_method = None
            if watsonx is not None:
                to_dict_method = getattr(watsonx, "to_dict", None)
            if to_dict_method is None:
                to_dict_method = getattr(embedding, "to_dict", None)
            if callable(to_dict_method):
                embedding_f = to_dict_method()
            else:
                raise VectorStoreSerializationError(
                    f"Cannot serialize embedding-function of type {type(embedding).__name__}; "
                    "expected `.watsonx_embed.to_dict()` or `.to_dict()`."
                )

        return {
            "connection_id": self._connection_id,
            "embedding_function": embedding_f,
            "table_name": self._table_name,
            **self._additional_kwargs,
            "datasource_type": str(self._datasource_type),
        }

    @classmethod
    def from_dict(
        cls, api_client: APIClient | None = None, data: dict | None = None
    ) -> "DB2VectorStore":
        """Creates ``DB2VectorStore`` using only a primitive data type dict.

        :param api_client: initialised APIClient used in vector store constructor, defaults to None
        :type api_client: APIClient, optional

        :param data: dict in schema like the ``to_dict()`` method
        :type data: dict

        :return: reconstructed DB2VectorStore
        :rtype: DB2VectorStore
        """
        d = copy.deepcopy(data) if isinstance(data, dict) else {}

        # Remove `datasource_type` if present
        d.pop("datasource_type", None)

        if "embeddings" in d:
            d.setdefault("embedding_function", d.pop("embeddings"))

        if "index_name" in d:
            d.setdefault("table_name", d.pop("index_name"))

        if "distance_metric" in d:
            d.setdefault("distance_strategy", d.pop("distance_metric"))

        d["embedding_function"] = BaseEmbeddings.from_dict(
            data=d.get("embedding_function", {}), api_client=api_client
        )

        return cls(api_client, **d)
