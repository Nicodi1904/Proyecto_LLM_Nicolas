#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import copy
import logging
from typing import Any, cast

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import (
    MissingExtension,
    VectorStoreSerializationError,
)

try:
    import elasticsearch
    from langchain_elasticsearch.vectorstores import ElasticsearchStore

except ImportError:
    raise MissingExtension("langchain_elasticsearch")

from elastic_transport import ConnectionTimeout
from langchain_core.documents import Document

from .es_utils import HybridStrategyElasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore(LangChainVectorStoreAdapter[ElasticsearchStore]):
    """Elasticsearch vector store client for a RAG pattern.

    Instantiates the vector store connection in the watsonx.ai environment and handles the necessary operations.
    The parameters given by the keyword arguments are used to instantiate the vector store client in their
    particular constructor. Those parameters might be parsed differently.

    :param api_client: api client is required if connecting by connection_id, defaults to None
    :type api_client: APIClient, optional

    :param connection_id: connection asset ID, defaults to None
    :type connection_id: str, optional

    :param vector_store: initialized langchain_elasticsearch vector store, defaults to None
    :type vector_store: langchain_elasticsearch.ElasticsearchStore, optional

    :param embeddings: default dense embeddings to be used, defaults to None
    :type embeddings: BaseEmbeddings, optional

    :param index_name: name of the vector database index, defaults to None
    :type index_name: str, optional

    :param kwargs: keyword arguments that will be directly passed to `langchain_elasticsearch.ElasticsearchStore` constructor
    :type kwargs: Any, optional

    .. note::

        For hybrid search (multi-vector search), if no ranker type is specified in strategy, a `weighted` reranker with default weights equal to 1 is used.
        For more details, see the `langchain-elasticsearch documentation <https://python.langchain.com/docs/integrations/vectorstores/elasticsearch/>`_ and
        `Elasticsearch documentation <https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#_combine_approximate_knn_with_other_features>`_.

    .. warning::

        The default retrieval strategy is the same as in ``langchain_elasticsearch.ElasticsearchStore``, i.e. when no strategy is specified the ``elasticsearch.helpers.vectorstore.DenseVectorStrategy``
        will be used (see `langchain-elasticsearch` `documentation <https://python.langchain.com/api_reference/elasticsearch/vectorstores/langchain_elasticsearch.vectorstores.ElasticsearchStore.html#langchain_elasticsearch.vectorstores.ElasticsearchStore>`_).

        Please note, that this strategy differ from the default one in ``ibm_watsonx.ai.foundation_models.extensions.rag.vector_stores.VectorStore``, where ``elasticsearch.helpers.vectorstore.DenseVectorScriptScoreStrategy`` is used. To ensure the same functionality
        when migrating from ``VectorStore`` to ``ElasticsearchVectorStore``, you may want to pass ``DenseVectorScriptScoreStrategy(distance=distance_metric)`` explicitly to ``ElasticsearchVectorStore`` constructor.


    **Example:**

    To connect, provide the connection asset ID.
    You can use custom embeddings to add and search documents.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import ElasticsearchVectorStore
        from ibm_watsonx_ai.foundation_models.embeddings import Embeddings

        credentials = Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"
                )

        api_client = APIClient(credentials)

        embedding = Embeddings(
                 model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
                 api_client=api_client
                 )

        vector_store = ElasticsearchVectorStore(
                api_client,
                connection_id='***',
                index_name='my_test_index',
                embeddings=embedding
            )

        vector_store.add_documents([
            {'content': 'document one content', 'metadata':{'url':'ibm.com'}},
            {'content': 'document two content', 'metadata':{'url':'ibm.com'}}
        ])

        vector_store.search('one', k=1)

    .. note::
        Optionally, like in LangChain, it is possible to use direct credentials to connect to Elastic Cloud.
        The keyword arguments can be used as direct params to LangChain's ``ElasticsearchStore`` constructor.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            ElasticsearchVectorStore,
            HybridStrategyElasticsearch,
            RetrievalOptions,
        )

        credentials = Credentials(api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com")

        api_client = APIClient(credentials)

        vector_store = ElasticsearchVectorStore(
            api_client,
            index_name="my_test_index",
            strategy=HybridStrategyElasticsearch(
                retrieval_strategies={RetrievalOptions.SPARSE: {"model_id": ".elser"}}
            ),
            cloud_id="***",
            api_key=IAM_API_KEY,
        )

        vector_store.add_documents(
            [
                {"content": "document one content", "metadata": {"url": "ibm.com"}},
                {"content": "document two content", "metadata": {"url": "ibm.com"}},
            ]
        )

        vector_store.search("one", k=1)

    .. note::
        To use hybrid search please specify multiple retrieval strategies in HybridStrategyElasticsearch.

    Example with weighted ranker.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            ElasticsearchVectorStore,
            HybridStrategyElasticsearch,
            RetrievalOptions,
        )

        credentials = Credentials(api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com")

        api_client = APIClient(credentials)

        vector_store = ElasticsearchVectorStore(
            api_client,
            connection_id=es_connection_id,
            index_name="my_test_index",
            strategy=HybridStrategyElasticsearch(
                retrieval_strategies={
                    RetrievalOptions.SPARSE: {"model_id": ".elser", "boost": 0.5},
                    RetrievalOptions.BM25: {"boost": 1},
                }
            ),
        )

        vector_store.add_documents(
            [
                {"content": "document one content", "metadata": {"url": "ibm.com"}},
                {"content": "document two content", "metadata": {"url": "ibm.com"}},
            ]
        )

        vector_store.search("one", k=1)


    Example with rrf ranker:

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            ElasticsearchVectorStore,
            HybridStrategyElasticsearch,
            RetrievalOptions,
        )

        credentials = Credentials(api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com")

        api_client = APIClient(credentials)

        vector_store = ElasticsearchVectorStore(
            api_client,
            connection_id=es_connection_id,
            index_name="my_test_index",
            strategy=HybridStrategyElasticsearch(
                retrieval_strategies={
                    RetrievalOptions.SPARSE: {"model_id": ".elser"},
                    RetrievalOptions.BM25: {},
                },
                use_rrf=True
                rrf_params={"k": 50}
            ),
        )

        vector_store.add_documents(
            [
                {"content": "document one content", "metadata": {"url": "ibm.com"}},
                {"content": "document two content", "metadata": {"url": "ibm.com"}},
            ]
        )

        vector_store.search("one", k=1)


    """

    def __init__(
        self,
        api_client: APIClient | None = None,
        *,
        connection_id: str | None = None,
        vector_store: ElasticsearchStore | None = None,
        index_name: str | None = None,
        embedding: BaseEmbeddings | None = None,
        **kwargs: Any,
    ) -> None:
        self._connection_id = connection_id
        self._client = api_client

        self._is_serializable = not bool(vector_store)

        # used in .to_dict method
        self._embedding = embedding
        self._index_name = index_name
        self._index_properties = kwargs
        ###

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
                    VectorStoreDataSourceType.ELASTICSEARCH,
                    {},
                )

            logger.info("Initializing vector store of type: %s", self._datasource_type)

            # overwrite text and vector field names set in langchain_elasticsearch
            if isinstance(
                (strategy := self._index_properties.get("strategy")),
                HybridStrategyElasticsearch,
            ):
                if "query_field" not in self._index_properties:
                    self._index_properties["query_field"] = strategy._text_field
                if "vector_query_field" not in self._index_properties:
                    self._index_properties["vector_query_field"] = (
                        strategy._dense_vector_field
                    )

            self._properties = {
                **connection_properties,
                **self._index_properties,
                "embedding": self._embedding,
                "index_name": index_name,
            }

            self._properties = VectorStoreConnector(
                self._properties
            )._get_elasticsearch_connection_params()
            vector_store = ElasticsearchStore(**self._properties)
        else:
            self._datasource_type = (
                VectorStoreConnector.get_type_from_langchain_vector_store(vector_store)
            )

        super().__init__(vector_store=vector_store)

    def get_client(self) -> ElasticsearchStore:
        """Get langchain_elasticsearch.ElasticsearchStore instance."""
        return super().get_client()

    def clear(self) -> None:
        """
        Clear index by removing all records.
        """
        es_vs = self.get_client()._store
        es = self.get_client().client
        try:
            es.delete_by_query(
                index=es_vs.index, body={"query": {"match_all": {}}}, refresh=True
            )
        except elasticsearch.NotFoundError:
            pass

    def count(self) -> int:
        """
        Count number of records in index.
        """
        es = self.get_client().client
        return es.count(index=self.get_client()._store.index)["count"]

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

        return self._fallback_add_documents(ids, docs, texts, metadatas, **kwargs)

    def _fallback_add_documents(
        self,
        ids: list[str],
        docs: list[Document],
        texts: list[str],
        metadatas: list[Any],
        chunk_size: int = 500,  # default set to 500
        text_embeddings: list[tuple[str, list[float]]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        stop_fallback = False

        bulk_kwargs: dict[str, Any] | None = kwargs.pop("bulk_kwargs", None)
        if bulk_kwargs is None:
            bulk_kwargs = {"chunk_size": chunk_size}
        elif bulk_kwargs is not None and bulk_kwargs.get("chunk_size") is None:
            bulk_kwargs["chunk_size"] = chunk_size
        else:
            stop_fallback = True

        if self._embedding and not text_embeddings:
            vectors = self._embedding.embed_documents(texts)  # type: ignore[union-attr]
            text_embeddings = [(text, vector) for text, vector in zip(texts, vectors)]

        try:
            if text_embeddings:
                return self._langchain_vector_store.add_embeddings(
                    text_embeddings=text_embeddings,  # type: ignore[arg-type]
                    metadatas=metadatas,
                    ids=ids,
                    bulk_kwargs=bulk_kwargs,
                    **kwargs,
                )
            else:
                return self._langchain_vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                    bulk_kwargs=bulk_kwargs,
                    **kwargs,
                )
        except ConnectionTimeout as e:
            if chunk_size <= 50 or stop_fallback:
                raise e
            return self._fallback_add_documents(
                ids=ids,
                docs=docs,
                texts=texts,
                metadatas=metadatas,
                chunk_size=50,
                text_embeddings=text_embeddings,
                **kwargs,
            )

    def to_dict(self) -> dict:
        """Serialize ``ElasticsearchVectorStore`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the `from_dict` initialization
        :rtype: dict

        :raises VectorStoreSerializationError: when instance is not serializable
        """
        if not self._is_serializable:
            raise VectorStoreSerializationError(
                "Serialization is not available when passing vector store instance in `ElasticsearchVectorStore` constructor."
            )

        strategy = self._index_properties.get("strategy")
        if (
            strategy is not None
            and not isinstance(
                strategy,
                HybridStrategyElasticsearch,
            )
        ) or (
            self._embedding is not None
            and not isinstance(self._embedding, BaseEmbeddings)
        ):
            raise VectorStoreSerializationError(
                (
                    "Serialization is allowed only for `HybridStrategyElasticsearch` strategy or when no strategy is provided and "
                    "dense embeddings is an instance of `ibm_watsonx_ai.foundation_models.embeddings.BaseEmbeddings`."
                )
            )
        data_dict = {
            "connection_id": self._connection_id,
            "embedding": (
                self._embedding.to_dict() if self._embedding is not None else None
            ),
            "index_name": self._index_name,
            **self._index_properties,
            "datasource_type": str(self._datasource_type),
        }
        if strategy is not None:
            data_dict["strategy"] = strategy.to_dict()

        return data_dict

    @classmethod
    def from_dict(
        cls, api_client: APIClient | None = None, data: dict | None = None
    ) -> "ElasticsearchVectorStore":
        """Creates ``ElasticsearchVectorStore`` using only a primitive data type dict.

        :param api_client: initialised APIClient used in vector store constructor, defaults to None
        :type api_client: APIClient, optional

        :param data: dict in schema like the ``to_dict()`` method
        :type data: dict

        :return: reconstructed VectorStore
        :rtype: VectorStore
        """
        d = copy.deepcopy(data) if isinstance(data, dict) else {}

        # Remove `datasource_type` if present
        d.pop("datasource_type", None)

        d["embedding"] = BaseEmbeddings.from_dict(
            data=d.get("embedding", {}), api_client=api_client
        )

        strategy_dict = d.get("strategy")
        if strategy_dict is not None:
            d["strategy"] = HybridStrategyElasticsearch.from_dict(data=strategy_dict)

        return cls(api_client, **d)
