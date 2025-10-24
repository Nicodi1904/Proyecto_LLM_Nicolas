#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import asyncio
import copy
from typing import Any, TypeAlias, cast

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import (
    InvalidValue,
    MissingExtension,
    VectorStoreSerializationError,
)

try:
    from langchain_core.embeddings import Embeddings as LCEmbeddings
    from langchain_milvus import Milvus
    from langchain_milvus.utils.sparse import (
        BaseSparseEmbedding as LCMilvusBaseSparseEmbedding,
    )

    from .milvus_utils import (
        DEFAULT_INDEX_PARAM,
        MilvusBM25BuiltinFunction,
        _LangchainEmbeddings,
    )

except ImportError as exc:
    raise MissingExtension(
        "langchain_milvus",
        reason="Please install `ibm-watsonx-ai` with flag `rag`: \n `pip install -U 'ibm-watsonx-ai[rag]'`",
    ) from exc

from langchain_core.documents import Document

try:
    from pymilvus import MilvusException
except ImportError:
    raise MissingExtension("pymilvus")

import logging

from ibm_watsonx_ai import APIClient

logger = logging.getLogger(__name__)

# Type Alias
EmbeddingType: TypeAlias = BaseEmbeddings | LCEmbeddings | LCMilvusBaseSparseEmbedding


class MilvusVectorStore(LangChainVectorStoreAdapter[Milvus]):
    """MilvusVectorStore vector store client for a RAG pattern.

    Instantiates the vector store connection in the watsonx.ai environment and handles the necessary operations.
    The parameters given by the keyword arguments are used to instantiate the vector store client in their
    particular constructor. Those parameters might be parsed differently.

    :param api_client: api client is required if connecting by connection_id, defaults to None
    :type api_client: APIClient, optional

    :param connection_id: connection asset ID, defaults to None
    :type connection_id: str, optional

    :param vector_store: initialized langchain_milvus vector store, defaults to None
    :type vector_store: langchain_milvus.Milvus, optional

    :param embedding_function: list of dense or sparse embedding function, defaults to None
    :type embedding_function: BaseEmbeddings | LCEmbeddings | LCMilvusBaseSparseEmbedding | list[BaseEmbeddings | LCEmbeddings | LCMilvusBaseSparseEmbedding], optional

    :param collection_name: name of the Milvus vector database collection, defaults to None
    :type collection_name: str, optional

    :param kwargs: keyword arguments that will be directly passed to `langchain_milvus.Milvus` constructor
    :type kwargs: Any, optional

    .. note::

        For hybrid search (multi-vector search), if no `ranker_type` is specified, a `weighted` reranker with default weights equal to 1 is used.
        For more details, see the `langchain_milvus` documentation https://python.langchain.com/docs/integrations/vectorstores/milvus/#hybrid-search.

    **Example:**

    To connect, provide the connection asset ID.
    You can use custom embeddings to add and search documents.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import MilvusVectorStore
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

        vector_store = MilvusVectorStore(
                api_client,
                connection_id='***',
                collection_name='my_test_collection',
                embedding_function=embedding
            )

        vector_store.add_documents([
            {'content': 'document one content', 'metadata':{'url':'ibm.com'}},
            {'content': 'document two content', 'metadata':{'url':'ibm.com'}}
        ])

        vector_store.search('one', k=1)

    .. note::
        To use hybrid search you need to pass several embedding function.

    Example with weighted ranker.

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            MilvusVectorStore,
            MilvusSpladeEmbeddingFunction
        )

        credentials = Credentials(api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com")

        api_client = APIClient(credentials)

        dense_embedding = Embeddings(
                 model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
                 api_client=api_client
                 )

        splade_func = MilvusSpladeEmbeddingFunction(model_name="naver/splade-cocondenser-selfdistil", device="cpu")

        vector_store = MilvusVectorStore(
            api_client,
            connection_id=es_connection_id,
            collection_name="my_test_collection",
            embedding_function=[dense_embedding, splade_func]
        )

        vector_store.add_documents(
            [
                {"content": "document one content", "metadata": {"url": "ibm.com"}},
                {"content": "document two content", "metadata": {"url": "ibm.com"}},
            ]
        )

        # `weighted` ranker
        vector_store.search("one", k=1, ranker_type="weighted", ranker_params={"weights": [0.0, 1.0])

        # `rrf` ranker
        vector_store.search("one", k=1, ranker_type="rrf", ranker_params={"k": 50)


    .. note::
        Please note that since Milvus v2.5 a full-text search can be used https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            MilvusVectorStore,
            MilvusBM25BuiltinFunction(
        )

        credentials = Credentials(api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com")

        api_client = APIClient(credentials)

        dense_embedding = Embeddings(
                 model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
                 api_client=api_client
                 )
        bm25_builtin_func = MilvusBM25BuiltinFunction()

        vector_store = MilvusVectorStore(
            api_client,
            connection_id=es_connection_id,
            collection_name="my_test_collection",
            embedding_function=dense_embedding,
            builtin_function=bm25_builtin_func,
        )

        vector_store.add_documents(
            [
                {"content": "document one content", "metadata": {"url": "ibm.com"}},
                {"content": "document two content", "metadata": {"url": "ibm.com"}},
            ]
        )

        # `weighted` ranker
        vector_store.search("one", k=1, ranker_type="weighted", ranker_params={"weights": [0.0, 1.0])

        # `rrf` ranker
        vector_store.search("one", k=1, ranker_type="rrf", ranker_params={"k": 50)


    """

    def __init__(
        self,
        api_client: APIClient | None = None,
        *,
        connection_id: str | None = None,
        vector_store: Milvus | None = None,
        embedding_function: EmbeddingType | list[EmbeddingType] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._connection_id = connection_id
        self._client = api_client

        self._is_serializable = not bool(vector_store)

        # For backward compatibility
        distance_metric = kwargs.pop("distance_metric", None)
        if distance_metric == "cosine":
            kwargs["index_params"] = DEFAULT_INDEX_PARAM

        self._embedding_function = embedding_function
        self._builtin_function = kwargs.pop("builtin_function", None)
        if isinstance(self._embedding_function, list):
            self._embedding_function = [
                (
                    _LangchainEmbeddings(embed_func)
                    if isinstance(embed_func, BaseEmbeddings)
                    and not isinstance(
                        embed_func, (LCEmbeddings, LCMilvusBaseSparseEmbedding)
                    )
                    else embed_func
                )
                for embed_func in self._embedding_function
            ]
        self._collection_name = collection_name
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
                    VectorStoreDataSourceType.MILVUS,
                    {},
                )

            logger.info("Initializing vector store of type: %s", self._datasource_type)

            self._properties = {
                **connection_properties,
                **self._additional_kwargs,
                "embedding_function": self._embedding_function,
                "collection_name": self._collection_name,
                "builtin_function": self._builtin_function,
            }

            self._properties = VectorStoreConnector(
                self._properties
            )._get_milvus_connection_params()
            vector_store = Milvus(**self._properties)
        else:
            self._datasource_type = (
                VectorStoreConnector.get_type_from_langchain_vector_store(vector_store)
            )

        self._embedding_function = cast(list, self._embedding_function)

        super().__init__(
            vector_store=vector_store,
        )

    def get_client(self) -> Milvus:
        """Get langchain_milvus.Milvus instance."""
        return super().get_client()

    def clear(self) -> None:
        """
        Clear collection by removing all records.
        """
        ids = self.get_client().get_pks("pk != ''")
        if ids:
            self.delete(ids)  # type: ignore[arg-type]

    def count(self) -> int:
        """
        Count number of records in collection.
        """
        ids = self.get_client().get_pks("pk != ''")
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

        # Repeat the logic from langchain_milvus.add_texts:
        # https://github.com/langchain-ai/langchain-milvus/blob/main/libs/milvus/langchain_milvus/vectorstores/milvus.py#L769

        match self._langchain_vector_store.embedding_func:
            case list() as func_list:
                embeddings_functions = func_list
            case None:
                embeddings_functions = []
            case _ as func:
                embeddings_functions = [func]

        embeddings = []
        for embedding_func in embeddings_functions:
            try:
                embeddings.append(embedding_func.embed_documents(texts))
            except NotImplementedError:
                embeddings.append([embedding_func.embed_query(x) for x in texts])  # type: ignore[arg-type]

        if isinstance(self._langchain_vector_store.embedding_func, list):
            transposed_embeddings: list | list[list] = [
                [embeddings[j][i] for j in range(len(embeddings))]
                for i in range(len(embeddings[0]))
            ]
        else:
            transposed_embeddings = embeddings[0] if len(embeddings) > 0 else []

        return self._fallback_add_documents(
            ids, docs, texts, metadatas, transposed_embeddings, **kwargs
        )

    async def add_documents_async(
        self, content: list[str] | list[dict] | list[Document], **kwargs: Any
    ) -> list[str]:
        """
        Embed documents and add to the vectorstore in asynchronous manner.

        :param content: Documents to add to the vectorstore.
        :type content: list[str] | list[dict] | list[langchain_core.documents.Document]

        :return: List of IDs of the added texts.
        :rtype: list[str]
        """

        return await asyncio.to_thread(
            self.add_documents,
            content,
            **kwargs,
        )

    def _fallback_add_documents(
        self,
        ids: list[str],
        docs: list[Document],
        texts: list[str],
        metadatas: list[Any],
        embeddings: list[Any],
        batch_size: int = 1024,  # default set to 1024
        **kwargs: Any,
    ) -> list[str]:
        if batch_size <= 0:
            raise InvalidValue(
                "batch_size",
                "`batch_size` reached 0 in fallback method for Milvus database. Either documents are too large or `batch_size` was set incorrectly.",
            )
        try:
            return self._upsert(
                ids=ids,
                docs=docs,
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                batch_size=batch_size,
                **kwargs,
            )
        except MilvusException as e:
            if (
                e.code == 65535
            ):  # handle MilvusException: (code=65535, message=Broker: Message size too large)
                return self._fallback_add_documents(
                    ids=ids,
                    docs=docs,
                    texts=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    batch_size=batch_size // 4,
                    **kwargs,
                )
            else:
                raise e

    def _upsert(
        self,
        ids: list[str],
        docs: list[Document],
        texts: list[str],
        metadatas: list[Any],
        embeddings: list[Any],
        **kwargs: Any,
    ) -> list[str]:
        """Upsert with custom ids.
        Based on Milvus LangChain upsert, but passes ids to add_documents.

        :param ids: list of ids for docs to upsert, defaults to None
        :type ids: list[str]

        :param docs: list of docs, defaults to None
        :type docs: list[Document]

        :return: list of added/upserted ids
        :rtype: list[str]
        """

        if docs is None or len(docs) == 0:
            return []

        if ids is not None and len(ids) and self.get_client().col is not None:
            try:
                self.delete(ids=ids)
            except MilvusException:
                pass

        try:
            return self.get_client().add_embeddings(  # type: ignore[attr-defined]
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                **kwargs,
            )
        except AttributeError:
            return self.get_client().add_documents(ids=ids, documents=docs, **kwargs)

    def to_dict(self) -> dict:
        """Serialize ``MilvusVectorStore`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict

        :raises VectorStoreSerializationError: when instance is not serializable
        """
        if not self._is_serializable:
            raise VectorStoreSerializationError(
                "Serialization is not available when passing vector store instance in `MilvusVectorStore` constructor."
            )

        if self._embedding_function is None:
            embedding_function: dict | list[dict] | None = None
        elif isinstance(self._embedding_function, list):
            embedding_function = []
            for embed_func in self._embedding_function:
                if isinstance(
                    embed_func,
                    (
                        BaseEmbeddings,
                        _LangchainEmbeddings,
                    ),
                ):
                    embedding_function.append(embed_func.to_dict())
                else:
                    raise VectorStoreSerializationError(
                        "Serialization is only available when 'embedding_function' are the instances of `ibm_watsonx_ai.foundation_models.embeddings.BaseEmbeddings`"
                    )
        elif isinstance(
            self._embedding_function,
            (BaseEmbeddings, _LangchainEmbeddings),
        ):
            embedding_function = [self._embedding_function.to_dict()]
        else:
            raise VectorStoreSerializationError(
                "Serialization is only available when 'embedding_function' is an instance of `ibm_watsonx_ai.foundation_models.embeddings.BaseEmbeddings`"
            )

        if self._builtin_function is None:
            builtin_function: dict | list[dict] | None = None
        elif isinstance(self._builtin_function, list):
            builtin_function = []
            for embed_func in self._builtin_function:
                if isinstance(embed_func, MilvusBM25BuiltinFunction):
                    builtin_function.append(embed_func.to_dict())
                else:
                    raise VectorStoreSerializationError(
                        "Serialization is only available when each element of 'builtin_function' is an instance of `MilvusBM25BuiltinFunction`"
                    )
        elif isinstance(self._builtin_function, MilvusBM25BuiltinFunction):
            builtin_function = [self._builtin_function.to_dict()]
        else:
            raise VectorStoreSerializationError(
                "Serialization is only available when 'builtin_function' is an instance of `MilvusBM25BuiltinFunction`"
            )

        return {
            "connection_id": self._connection_id,
            "embedding_function": embedding_function,
            "collection_name": self._collection_name,
            "builtin_function": builtin_function,
            **self._additional_kwargs,
            "datasource_type": str(self._datasource_type),
        }

    @classmethod
    def from_dict(
        cls, api_client: APIClient | None = None, data: dict | None = None
    ) -> "MilvusVectorStore":
        """Creates ``MilvusVectorStore`` using only a primitive data type dict.

        :param api_client: initialised APIClient used in vector store constructor, defaults to None
        :type api_client: APIClient, optional

        :param data: dict in schema like the ``to_dict()`` method
        :type data: dict

        :return: reconstructed MilvusVectorStore
        :rtype: MilvusVectorStore
        """
        d = copy.deepcopy(data) if isinstance(data, dict) else {}

        # Remove `datasource_type` if present
        d.pop("datasource_type", None)

        d["embedding_function"] = (
            [
                BaseEmbeddings.from_dict(data=embed_func_dict, api_client=api_client)
                for embed_func_dict in embedding_function
            ]
            if (embedding_function := d.get("embedding_function", [])) is not None
            else None
        )

        d["builtin_function"] = (
            [
                BaseEmbeddings.from_dict(data=embed_func_dict, api_client=api_client)
                for embed_func_dict in builtin_function
            ]
            if (builtin_function := d.get("builtin_function", [])) is not None
            else None
        )

        return cls(api_client, **d)
