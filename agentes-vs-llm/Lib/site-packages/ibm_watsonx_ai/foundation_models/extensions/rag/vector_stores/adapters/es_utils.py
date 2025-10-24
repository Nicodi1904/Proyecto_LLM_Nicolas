#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any, Optional

from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    import elasticsearch
    from elasticsearch.helpers.vectorstore import (
        RetrievalStrategy,
    )
    from elasticsearch.helpers.vectorstore._sync._utils import model_must_be_deployed
except ImportError:
    raise MissingExtension("langchain_elasticsearch")

TEXT_FIELD = "text_field"


class RetrievalOptions:
    """Retrieval options to be used when conducting hybrid search."""

    DENSE = "dense"
    SPARSE = "sparse"
    BM25 = "bm25"


# Based on the https://github.com/elastic/elasticsearch-py/issues/2630 IBM Research investigation
class HybridStrategyElasticsearch(RetrievalStrategy):
    """Hybrid strategy to be used in `ElasticsearchVectorStore` to take advantage of hybrid search.

    :param retrieval_strategies: mapping containing retrieval type and its properties
    :type retrieval_strategies: dict[str, dict[str, Any]]

    :param use_rrf: whether to use Reciprocal Rank Fusion (rrf) ranker when combining multiple results search in hybrid approach.
                    For more details, please visit https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html, defaults to False
    :type use_rrf: bool, optional

    :param rrf_params: rrf method's parameters, default to None
    :type rrf_params: dict, optional

    :param text_field: text field name, default to `text_field`
    :type text_field: str, optional

    **Example:**

    When no ranker method is explicitly specified, the weighted ranker is used with all weights equal to 1.
    To change the weight for particular strategy add `boost` field to retrieval type settings.

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            HybridStrategyElasticsearch,
            RetrievalOptions,
        )


        strategy=HybridStrategyElasticsearch(
            retrieval_strategies={
                RetrievalOptions.SPARSE: {"model_id": ".elser", "boost": 0.5},
                RetrievalOptions.BM25: {"boost": 1},
            }
        )


    Example with rrf ranker:

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import (
            HybridStrategyElasticsearch,
            RetrievalOptions,
        )



        strategy=HybridStrategyElasticsearch(
            retrieval_strategies={
                RetrievalOptions.SPARSE: {"model_id": ".elser"},
                RetrievalOptions.BM25: {},
            },
            use_rrf=True
            rrf_params={"k": 50}
        )

    """

    _sparse_vector_field = "sparse_vector"
    _dense_vector_field = "dense_vector"
    _tokens_field = "tokens"
    _sparse_model_id = ".elser_model_2"
    _dense_model_id = None

    def __init__(
        self,
        retrieval_strategies: dict[str, dict[str, Any]],
        use_rrf: bool = False,
        rrf_params: dict | None = None,
        text_field: str = TEXT_FIELD,
    ):
        self._retrieval_strategies = retrieval_strategies
        self._text_field = text_field

        self._dense_num_dimensions = None

        if RetrievalOptions.DENSE in self._retrieval_strategies:
            dense_strategy_config = self._retrieval_strategies[RetrievalOptions.DENSE]
            self._dense_num_dimensions = dense_strategy_config.get("num_dimensions")

            self._dense_model_id = dense_strategy_config.get("model_id")
            if (vector_field := dense_strategy_config.get("vector_field")) is not None:
                self._dense_vector_field = vector_field

        if RetrievalOptions.SPARSE in self._retrieval_strategies:
            self._pipeline_name = f"{self._sparse_model_id}_sparse_embedding"
            sparse_strategy_config = self._retrieval_strategies[RetrievalOptions.SPARSE]
            if (model_id := sparse_strategy_config.get("model_id")) is not None:
                self._sparse_model_id = model_id
            if (vector_field := sparse_strategy_config.get("vector_field")) is not None:
                self._sparse_vector_field = vector_field

        if RetrievalOptions.BM25 in self._retrieval_strategies:
            bm25_strategy_config = self._retrieval_strategies[RetrievalOptions.BM25]
            if (bm25_text_field := bm25_strategy_config.get("text_field")) is not None:
                self._text_field = bm25_text_field

        if not use_rrf and (
            any(
                "boost" not in strategy
                for _, strategy in self._retrieval_strategies.items()
            )
            and len(self._retrieval_strategies) != 1
        ):
            raise ValueError(
                "Either all strategies have assigned boost (aka weight) or none."
            )

        self.rrf = rrf_params if use_rrf else None

    def before_index_creation(
        self, *, client: elasticsearch.Elasticsearch, text_field: str, vector_field: str
    ) -> None:
        if RetrievalOptions.DENSE in self._retrieval_strategies:
            if self._dense_model_id:
                model_must_be_deployed(client, self._dense_model_id)

        if RetrievalOptions.SPARSE in self._retrieval_strategies:
            model_must_be_deployed(client, self._sparse_model_id)

            # Create a pipeline for the model
            client.ingest.put_pipeline(
                id=self._pipeline_name,
                description="Embedding pipeline for Python VectorStore",
                processors=[
                    {
                        "inference": {
                            "model_id": self._sparse_model_id,
                            "target_field": self._sparse_vector_field,
                            "field_map": {self._text_field: "text_field"},
                            "inference_config": {
                                "text_expansion": {"results_field": self._tokens_field}
                            },
                        }
                    }
                ],
            )

    def es_mappings_settings(
        self, *, text_field: str, vector_field: str, num_dimensions: Optional[int]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mappings: dict = {"properties": {}}
        settings: dict = {}

        if RetrievalOptions.DENSE in self._retrieval_strategies:
            dense_mappings = {
                "properties": {
                    self._dense_vector_field: {
                        "type": "dense_vector",
                        "dims": num_dimensions or self._dense_num_dimensions,
                        "index": True,
                        "similarity": self._retrieval_strategies["dense"].get(
                            "distance", "cosine"
                        ),
                    },
                }
            }
            mappings["properties"].update(dense_mappings["properties"])
            # No need to update settings

        if RetrievalOptions.SPARSE in self._retrieval_strategies:
            sparse_mappings = {
                "properties": {
                    self._sparse_vector_field: {
                        "properties": {self._tokens_field: {"type": "rank_features"}}
                    }
                }
            }
            sparse_settings = {"default_pipeline": self._pipeline_name}
            mappings["properties"].update(sparse_mappings["properties"])
            settings.update(sparse_settings)

        if RetrievalOptions.BM25 in self._retrieval_strategies:
            strategy_configs = self._retrieval_strategies["bm25"]
            similarity_name = "custom_bm25"
            bm25_mappings = {
                "properties": {
                    self._text_field: {
                        "type": "text",
                        "similarity": similarity_name,
                    },
                },
            }
            bm25 = {
                "type": "BM25",
            }
            if "k1" in strategy_configs:
                bm25["k1"] = strategy_configs["k1"]
            if "b" in strategy_configs:
                bm25["b"] = strategy_configs["b"]
            bm25_settings = {
                "similarity": {
                    similarity_name: bm25,
                }
            }

            mappings["properties"].update(bm25_mappings["properties"])
            settings.update(bm25_settings)

        return mappings, settings

    def es_query(
        self,
        *,
        query: Optional[str],
        query_vector: Optional[list[float]],
        text_field: str,
        vector_field: str,
        k: int,
        num_candidates: int,
        filter: list[dict[str, Any]] = [],
    ) -> dict[str, Any]:
        standard_query = {}

        if RetrievalOptions.DENSE in self._retrieval_strategies:
            knn_query = {
                "filter": filter,
                "field": self._dense_vector_field,
                "k": k,
                "num_candidates": num_candidates,
            }

            if query_vector is not None:
                knn_query["query_vector"] = query_vector
            else:
                # Inference in Elasticsearch.
                knn_query["query_vector_builder"] = {
                    "text_embedding": {
                        "model_id": self._dense_model_id,
                        "model_text": query,
                    }
                }

        if RetrievalOptions.SPARSE in self._retrieval_strategies:
            sparse_query = {
                "text_expansion": {
                    f"{self._sparse_vector_field}.{self._tokens_field}": {
                        "model_id": self._sparse_model_id,
                        "model_text": query,
                    }
                },
            }
            if "query" not in standard_query:
                standard_query.update(
                    {
                        "query": {
                            "bool": {
                                "must": [sparse_query],
                                "filter": filter,
                            }
                        }
                    }
                )
            else:
                standard_query["query"]["bool"]["must"].append(sparse_query)

        if RetrievalOptions.BM25 in self._retrieval_strategies:
            bm25_must_query = {
                "match": {
                    self._text_field: {
                        "query": query,
                    }
                },
            }
            if "query" not in standard_query:
                standard_query.update(
                    {
                        "query": {
                            "bool": {
                                "must": [bm25_must_query],
                                "filter": filter,
                            }
                        }
                    }
                )
            else:
                standard_query["query"]["bool"]["must"].append(bm25_must_query)

        if self.rrf:
            rrf_options = {}
            if isinstance(self.rrf, dict):
                if rank_constant := (
                    self.rrf.get("rank_constant") or self.rrf.get("k")
                ):
                    rrf_options["rank_constant"] = rank_constant
                if "rank_window_size" in self.rrf:
                    rrf_options["rank_window_size"] = self.rrf["rank_window_size"]

            retrievers: list[dict[str, Any]] = [
                {"standard": standard_query},
            ]
            if RetrievalOptions.DENSE in self._retrieval_strategies:
                retrievers.append(
                    {"knn": knn_query},
                )
            query_body = {
                "retriever": {
                    "rrf": {
                        "retrievers": retrievers,
                        **rrf_options,
                    },
                },
            }
            return query_body
        else:
            final_query: dict = {}

            if RetrievalOptions.DENSE in self._retrieval_strategies:
                knn_query["boost"] = self._retrieval_strategies["dense"]["boost"]
                final_query |= {"knn": knn_query}
            if RetrievalOptions.SPARSE in self._retrieval_strategies:
                must_query = standard_query["query"]["bool"]["must"]
                if "text_expansion" in must_query[0]:
                    must_query[0]["text_expansion"][
                        f"{self._sparse_vector_field}.{self._tokens_field}"
                    ]["boost"] = self._retrieval_strategies["sparse"]["boost"]
                else:
                    must_query[1]["text_expansion"][
                        f"{self._sparse_vector_field}.{self._tokens_field}"
                    ]["boost"] = self._retrieval_strategies["sparse"]["boost"]
            if RetrievalOptions.BM25 in self._retrieval_strategies:
                must_query = standard_query["query"]["bool"]["must"]
                if "match" in must_query[0]:
                    must_query[0]["match"][self._text_field]["boost"] = (
                        self._retrieval_strategies["bm25"]["boost"]
                    )
                else:
                    must_query[1]["match"][self._text_field]["boost"] = (
                        self._retrieval_strategies["bm25"]["boost"]
                    )

            return final_query | standard_query

    def needs_inference(self) -> bool:
        return (
            RetrievalOptions.DENSE in self._retrieval_strategies
            and not self._dense_model_id
        )

    def to_dict(self) -> dict:
        """Serialize ``HybridStrategyElasticsearch`` into a dict that allows reconstruction using the ``from_dict`` class method.

        :return: dict for the from_dict initialization
        :rtype: dict
        """
        raw_data = {
            "retrieval_strategies": self._retrieval_strategies,
            "use_rrf": bool(self.rrf),
            "rrf_params": self.rrf,
        }

        if self._text_field is not TEXT_FIELD:
            raw_data["text_field"] = self._text_field

        return raw_data

    @classmethod
    def from_dict(cls, data: dict) -> "HybridStrategyElasticsearch":
        """Creates ``HybridStrategyElasticsearch`` using only a primitive data type dict.

        :param data: dict in schema like the ``to_dict()`` method
        :type data: dict

        :return: reconstructed HybridStrategyElasticsearch
        :rtype: HybridStrategyElasticsearch
        """
        return cls(**data)
