#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any

from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_milvus.function import BM25BuiltInFunction
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.wml_client_error import MissingExtension

__all__ = ["MilvusBM25BuiltinFunction", "MilvusSpladeEmbeddingFunction"]

DEFAULT_INDEX_PARAM = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}


class _LangchainEmbeddings(LCEmbeddings):
    """Helper class to allow passing `ibm_watsonx_ai.foundation_models.embeddings.BaseEmbeddings` to langchain_milvus"""

    def __init__(self, embeddings: BaseEmbeddings) -> None:
        super().__init__()

        self._embedding_func: BaseEmbeddings = embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embedding_func.embed_query(text=text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_func.embed_documents(texts=texts)

    def to_dict(self) -> dict:
        return self._embedding_func.to_dict()


class MilvusBM25BuiltinFunction(BM25BuiltInFunction):
    """
    Milvus BM25 built-in function.

    Wrapper for `langchain_milvus.BM25BuiltinFunction` that can be used together with MilvusVectorStore in RAGPattern.

    See:
    https://milvus.io/docs/full-text-search.md
    """

    def __init__(
        self,
        input_field_names: str = "text",
        output_field_names: str = "sparse",
        analyzer_params: dict[Any, Any] | None = None,
        enable_match: bool = False,
        function_name: str | None = None,
    ) -> None:
        super().__init__(
            input_field_names=input_field_names,
            output_field_names=output_field_names,
            enable_match=enable_match,
            function_name=function_name,
            analyzer_params=analyzer_params,
        )

    def to_dict(self) -> dict:
        """Serialize ``MilvusBM25BuiltinFunction`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """
        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__module__,
            "input_field_names": self._function._input_field_names,  # type: ignore[union-attr]
            "output_field_names": self._function._output_field_names,  # type: ignore[union-attr]
            "analyzer_params": self.analyzer_params,
            "enable_match": self.enable_match,
            "function_name": self._function._name,  # type: ignore[union-attr]
        }


class MilvusSpladeEmbeddingFunction(BaseSparseEmbedding, BaseEmbeddings):
    """Sparse embedding model based on SPLADE embedding.

    This class uses the one of the SPLADE model to implement sparse vector embedding.

     .. note::
        This model requires pymilvus[model] to be installed.
        `pip install pymilvus[model]`

    For more information please refer to: https://milvus.io/docs/embed-with-splade.md
    """

    def __init__(
        self, model_name: str = "naver/splade-cocondenser-ensembledistil", **kwargs: Any
    ) -> None:
        try:
            from pymilvus import model
        except ImportError:
            raise MissingExtension("pymilvus[model]")

        self._splade_ef = model.sparse.SpladeEmbeddingFunction(
            model_name=model_name, **kwargs
        )

    @staticmethod
    def _sparse_to_dict(sparse_array: Any) -> dict[int, float]:
        """Based on the implementation of `langchain_milvus.utils.sparse.BM25SparseEmbedding._sparse_to_dict"""
        row_indices, col_indices = sparse_array.nonzero()
        non_zero_values = sparse_array.data
        result_dict = {}
        for col_index, value in zip(col_indices, non_zero_values):
            result_dict[col_index] = value
        return result_dict

    def embed_documents(self, texts: list[str]) -> list[dict[int, float]]:  # type: ignore[override]
        """Embed search docs."""
        sparse_arrays = self._splade_ef.encode_documents(texts)
        return [
            MilvusSpladeEmbeddingFunction._sparse_to_dict(sparse_array)
            for sparse_array in sparse_arrays
        ]

    def embed_query(self, query: str) -> dict[int, float]:  # type: ignore[override]
        """Embed query text."""
        return self.embed_documents([query])[0]

    def to_dict(self) -> dict:
        """Serialize ``MilvusSpladeEmbeddingFunction`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """
        class_data = super().to_dict()

        return class_data | {
            "query_instruction": self._splade_ef.query_instruction,
            "doc_instruction": self._splade_ef.doc_instruction,
            "k_tokens_query": self._splade_ef.k_tokens_query,
            "k_tokens_document": self._splade_ef.k_tokens_document,
            **self._splade_ef._model_config,
        }
