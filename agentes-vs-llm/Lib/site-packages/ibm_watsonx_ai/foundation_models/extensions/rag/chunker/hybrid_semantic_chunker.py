#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import re
from functools import reduce
from typing import Any, Callable, Sequence, TypeVar, cast

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.base_chunker import (
    BaseChunker,
)
from ibm_watsonx_ai.wml_client_error import (
    ApiRequestFailure,
    HybridSemanticChunkerException,
)

ChunkType = TypeVar("ChunkType", bound=Document)
DEFAULT_MIN_BREAKPOINT_CHUNK_SIZE = 100
DEFAULT_TFIDF_BUFFER_SIZE = 3
DEFAULT_EMBEDDING_BUFFER_SIZE = 3
DEFAULT_TFIDF_WEIGHT = 0.5
DEFAULT_EMBEDDING_WEIGHT = 0.5
ABBREVIATIONS = [
    "e.g.",
    "i.e.",
    "et al.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Prof.",
    "Fig.",
    "Inc.",
    "St.",
    "Jr.",
    "Sr.",
    "vs.",
    "etc.",
]


class HybridSemanticChunker(BaseChunker[Document]):
    """Chunker which uses similarity between inner segments of text to find optimal breakpoints.

    .. note::
        Added in 1.3.25

    :param embeddings: embeddings to be used to generate dense vectors
    :type embeddings: Embeddings

    :param chunk_size: approximate chunk size
    :type chunk_size: int

    :param allowed_chunk_size_deviation: specifies the fraction by which each chunk's size may vary
        from the target chunk_size
    :type allowed_chunk_size_deviation: float

    :param kwargs: additional chunker parameters:
        - tfidf_buffer_size: Number of breakpoint chunks to the left or right of each potential breakpoint
        used to generate TF-IDF vectors for similarity analysis, defaults to 5.
        - embedding_buffer_size: Number of breakpoint chunks to the left or right of each potential breakpoint
        used to generate embeddings for similarity analysis, defaults to 5.
        - tfidf_weight: Weight of tfidf vectors representations in similarity analysis, defaults to 0.5.
        - embedding_weight: Weight of embeddings in similarity analysis, defaults to 0.5.
    :type kwargs: dict

    Example:

    .. code-block:: python

        chunker = HybridSemanticChunker(embedding=embeddings)
        chunker.split_documents()

    or with vectors precomputing:

    .. code-block:: python

        chunker = HybridSemanticChunker(embedding=embeddings)
        chunker.precompute_vectors()
        chunker.get_chunks()

    """

    def __init__(
        self,
        embeddings: BaseEmbeddings,
        chunk_size: int = 1024,
        allowed_chunk_size_deviation: float = 1,
        **kwargs: Any,
    ):
        self.embeddings = embeddings
        self.tfidf_buffer_size = kwargs.get(
            "tfidf_buffer_size", DEFAULT_TFIDF_BUFFER_SIZE
        )
        self.embedding_buffer_size = kwargs.get(
            "embedding_buffer_size", DEFAULT_EMBEDDING_BUFFER_SIZE
        )
        self.chunk_size = chunk_size
        self.allowed_chunk_size_deviation = allowed_chunk_size_deviation
        self.min_breakpoint_chunk_size = DEFAULT_MIN_BREAKPOINT_CHUNK_SIZE
        self.tfidf_weight = kwargs.get("tfidf_weight", DEFAULT_TFIDF_WEIGHT)
        self.embedding_weight = kwargs.get("embedding_weight", DEFAULT_EMBEDDING_WEIGHT)

        self.documents: list[Document] = []
        self.too_short_docs: list[Document] = []
        self.docs_contents: list[str] = []
        self.docs_metadata: list[dict] = []
        self.breakpoint_chunks: list[list[str]] = []

        self.vectorizer = TfidfVectorizer()
        self.vectors: dict[str, list[dict]] = {}

    def precompute_vectors(self, documents: Sequence[Document], **kwargs: Any) -> None:
        """
        Performs an initial split using sentence_split_regex to identify potential breakpoints,
        then computes  and embedding vector representations of the text segments
        between breakpoints for semantic similarity analysis.
        This function is useful for experimenting with different chunking parameters efficiently,
        allowing to avoid recomputing vectors for the same input documents.

        :param documents: sequence of documents to perform chunking on
        :type documents: Sequence[Document]
        """
        self.documents = []
        self.too_short_docs = []
        for d in documents:
            if len(d.page_content) < self.chunk_size:
                d.metadata["sequence_number"] = 0
                self.too_short_docs.append(d)
            else:
                self.documents.append(d)

        if len(self.documents) > 0:
            self.docs_contents = [doc.page_content for doc in self.documents]
            self.docs_metadata = [doc.metadata for doc in self.documents]

            self.breakpoint_chunks = self._breakpoints_split()
            self.vectorizer.fit(self.docs_contents)
            self.vectors = {
                "tfidf": self._get_vectors(
                    vectors_fn=self._get_tfidf_vectors,
                    buffer_size=self.tfidf_buffer_size,
                ),
            }
            if kwargs.get("generate_embeddings", True):
                self.vectors["embedding"] = self._get_vectors(
                    vectors_fn=self._get_embeddings,
                    buffer_size=self.embedding_buffer_size,
                )

    def split_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> list[Document]:
        """
        Executes the full chunking process, including vector computation, similarity analysis,
        and selection of optimal breakpoints.

        :param documents: sequence of documents to perform chunking on
        :type documents: Sequence[Document]

        :param kwargs: chunking parameters
        :type kwargs: Any

        :return: list of chunks
        :rtype: list[Document]
        """
        self.precompute_vectors(
            documents, generate_embeddings=self.embedding_weight != 0
        )
        return self.get_chunks(**kwargs)

    def get_chunks(
        self,
        chunk_size: int | None = None,
        allowed_chunk_size_deviation: float | None = None,
        tfidf_weight: float | None = None,
        embedding_weight: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Performs similarity analysis on vector representations of the texts between potential breakpoints
        and identifies the optimal ones.

        :param chunk_size: approximate chunk size
        :type chunk_size: int, optional

        :param allowed_chunk_size_deviation: specifies the percentage by which each chunk's size may vary
            from the target chunk_size
        :type allowed_chunk_size_deviation: float, optional

        :param tfidf_weight: weight of tfidf vectors representations in similarity analysis
        :type tfidf_weight: float, optional

        :param embedding_weight: weight of embeddings in similarity analysis
        :type embedding_weight: float, optional

        :return: list of chunks
        :rtype: list[Document]
        """
        if len(self.documents) == 0:
            if self.too_short_docs:
                return self.too_short_docs
            else:
                raise HybridSemanticChunkerException(
                    "No documents to split, run precompute_vectors first or use split_documents instead."
                )

        if chunk_size:
            self.chunk_size = chunk_size

        allowed_chunk_size_deviation = (
            allowed_chunk_size_deviation or self.allowed_chunk_size_deviation
        )
        tfidf_weight = tfidf_weight or self.tfidf_weight
        embedding_weight = embedding_weight or self.embedding_weight

        similarities = self._get_similarities(
            weights={
                "embedding": embedding_weight,
                "tfidf": tfidf_weight,
            }
        )

        chunks = []
        for bp_chunks, sims, metadata in zip(
            self.breakpoint_chunks, similarities, self.docs_metadata
        ):
            chunks_strings = self._get_chunks_for_single_doc(
                bp_chunks, sims, self.chunk_size, allowed_chunk_size_deviation, **kwargs
            )
            chunks_docs = [
                Document(page_content=c, metadata=metadata | {"sequence_number": i})
                for i, c in enumerate(chunks_strings)
            ]
            chunks.extend(chunks_docs)

        return [*chunks, *self.too_short_docs]

    def _get_chunks_for_single_doc(
        self,
        bp_chunks: list[str],
        similarities: np.ndarray,
        chunk_size: int,
        allowed_chunk_size_deviation: float,
        chunk_overlap: int = 0,
    ) -> list[str]:
        """
        Generates chunks for a single document.

        :param bp_chunks: document content split at potential breakpoints
        :type bp_chunks: list[str]

        :param similarities: similarities of the windows of bp_chunks at each potential breakpoint
        :type similarities: np.ndarray

        :param chunk_size: approximate chunk size
        :type chunk_size: int

        :param allowed_chunk_size_deviation: specifies the percentage by which each chunk's size may vary
            from the target chunk_size
        :type allowed_chunk_size_deviation: float

        :params chunk_overlap: chunk overlap for recursive split if chunk is too big
        :type chunk_overlap: int

        :return: list of chunks contents
        :rtype: list[str]
        """
        local_mins = self._find_local_minima(similarities)

        sorted_local_mins = sorted(local_mins, key=lambda m: similarities[m])
        breakpoints: list[int] = []
        for minimum in sorted_local_mins:
            distances = [abs(fmin - minimum) for fmin in breakpoints]

            minimal_allowed_distance = int(
                self.chunk_size / self._mean_breakpoint_chunk_size
            )
            if not any(d < minimal_allowed_distance for d in distances):
                breakpoints.append(minimum)
        breakpoints = sorted(set(int(b) for b in breakpoints if 0 < b < len(bp_chunks)))

        full_breakpoints = [0, *breakpoints, len(bp_chunks)]
        semantic_chunks = []
        for start, end in zip(full_breakpoints[:-1], full_breakpoints[1:]):
            bp_start = start + 1 if start else 0
            bp_end = end + 1

            chunk = " ".join(bp_chunks[bp_start:bp_end])
            semantic_chunks.append(chunk)

        max_chunk_size = (1 + allowed_chunk_size_deviation) * chunk_size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = []
        for chunk in semantic_chunks:
            if len(chunk) > max_chunk_size:
                chunks.extend(splitter.split_text(chunk))
            else:
                chunks.append(chunk)

        return self._merge_too_small_texts(chunks, self._minimal_chunk_size)

    @staticmethod
    def _find_local_minima(values: np.ndarray) -> np.ndarray:
        """
        Finds local minimums in the list of numeric values.

        :param values: list of numeric values
        :type values: np.ndarray

        :return: indices of local minimums
        :rtype: np.ndarray
        """
        minima_mask = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
        minima_indices = np.where(minima_mask)[0] + 1
        return minima_indices

    @property
    def _minimal_chunk_size(self) -> int:
        """Minimal chunk size to avoid too small chunks."""
        return int(
            self.chunk_size - (self.chunk_size * self.allowed_chunk_size_deviation / 2)
        )

    @property
    def _maximal_chunk_size(self) -> int:
        """Maximal chunk size to avoid too big chunks."""
        return int((1 + self.allowed_chunk_size_deviation) * self.chunk_size)

    @property
    def _mean_breakpoint_chunk_size(self) -> int:
        return int(
            np.mean([len(s) for bp_chunks in self.breakpoint_chunks for s in bp_chunks])
        )

    def _get_tfidf_vectors(self, texts: str) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings from a sequence of texts.

        :param texts: list of texts to generate vectors from
        :type texts: list[str]

        :return: list of embeddings generated from texts
        :rtype: np.ndarray
        """
        try:
            return np.array(self.embeddings.embed_documents(texts))
        except ApiRequestFailure as e:
            raise HybridSemanticChunkerException(
                f"Error during generating embeddings: {e}"
            ) from e

    def _get_vectors(self, vectors_fn: Callable, buffer_size: int) -> list[dict]:
        """
        Creates a list of breakpoint chunks windows by taking buffer_size breakpoint chunks
        to the left and right of each potential breakpoint.
        Then applies vectors_fn to generate vector representations for the contents of these windows.
        Flattens the list of windows and regenerates the vectors structure to ensure vectors_fn is invoked only once.

        :param vectors_fn: function that generates vector representations from the windows of chunks
        :type vectors_fn: Callable

        :param buffer_size: number of breakpoint chunks on the left and right of potential breakpoint in each window
        :type buffer_size: int

        :return: list of vectors generated from the windows of breakpoint chunks
        :rtype: list[dict]
        """
        windows = self._get_windows(buffer_size)

        flat_windows: list[str] = reduce(
            lambda acc, d: acc + d["left"] + d["right"], windows, []
        )
        flat_vectors = vectors_fn(flat_windows)
        vectors = []
        i = 0
        for doc in windows:
            left_len = len(doc["left"])
            right_len = len(doc["right"])
            vectors.append(
                {
                    "left": flat_vectors[i : i + left_len],
                    "right": flat_vectors[i + left_len : i + left_len + right_len],
                }
            )
            i += left_len + right_len

        return vectors

    def _get_windows(self, buffer_size: int) -> list[dict]:
        """
        Generates windows of breakpoint chunks to generate vectors representations from.

        :param buffer_size: number of breakpoint chunks on the left and on the right from potential breakpoint
        :type buffer_size: int

        :return: lost of breakpoint chunks windows
        :rtype: list[dict]
        """
        buffer_size_as_text_length = self._mean_breakpoint_chunk_size * buffer_size
        windows = []
        for bp_chunks in self.breakpoint_chunks:
            len_chunks = len(bp_chunks)

            doc_lc = []
            doc_rc = []

            for i in range(1, len_chunks):
                start_idx = max(0, i - buffer_size)
                left_bp_chunks = bp_chunks[start_idx:i]
                while (
                    len(left_bp_chunks) > 1
                    and sum(len(c) for c in left_bp_chunks) > buffer_size_as_text_length
                ):
                    left_bp_chunks = left_bp_chunks[1:]
                lc = " ".join(left_bp_chunks)

                end_idx = min(len_chunks, i + buffer_size)
                right_bp_chunks = bp_chunks[i:end_idx]
                while (
                    len(right_bp_chunks) > 1
                    and sum(len(c) for c in right_bp_chunks)
                    > buffer_size_as_text_length
                ):
                    right_bp_chunks = right_bp_chunks[:-1]
                rc = " ".join(right_bp_chunks)

                doc_lc.append(lc)
                doc_rc.append(rc)

            windows.append({"left": doc_lc, "right": doc_rc})

        return windows

    def _get_similarities(self, weights: dict[str, float]) -> list:
        """
        Computes similarities between vectors generated from windows of breakpoint chunks,
        separately for tfidf and embedding vectors. Then uses MinMaxScaler to normalize both distributions
        and sums them using provided weights.

        :param weights: tfidf vector and embedding weight in final similarity score
        :type weights: dict[str, float]

        :return: list of similarities
        :type: list[float]
        """
        scaler = MinMaxScaler()

        tfidf_similarities = []
        for tfidf_vectors in self.vectors["tfidf"]:
            tfidf_sims = np.sum(tfidf_vectors["left"] * tfidf_vectors["right"], axis=1)
            tfidf_normalized = scaler.fit_transform(
                np.array(tfidf_sims).reshape(-1, 1)
            ).flatten()
            tfidf_similarities.append(tfidf_normalized)

        embedding_similarities = []
        if self.vectors.get("embedding"):
            for embedding_vectors in self.vectors["embedding"]:
                embedding_sims = np.diagonal(
                    cosine_similarity(
                        embedding_vectors["left"], embedding_vectors["right"]
                    )
                )
                embedding_normalized = scaler.fit_transform(
                    np.array(embedding_sims).reshape(-1, 1)
                ).flatten()
                embedding_similarities.append(embedding_normalized)

        if not embedding_similarities:
            return tfidf_similarities
        else:
            similarities = []
            for tfidf_sims, embedding_sims in zip(
                tfidf_similarities, embedding_similarities
            ):
                similarities.append(
                    weights["tfidf"] * tfidf_sims
                    + weights["embedding"] * embedding_sims
                )
            return similarities

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text on sentences.

        :param text: text to split
        :type text: str

        :return: text splitted on sentences.
        :rtype: list[str]
        """
        placeholder = "__DOT__"
        for abbr in ABBREVIATIONS:
            safe_abbr = abbr.replace(".", placeholder)
            text = text.replace(abbr, safe_abbr)

        # Split after ., !, or ? followed by space or \n
        # Split on double or more newlines (\n\n+)
        # Split before bullet points (-, *, •) only if preceded by a newline
        # Split before markdown headers (# Header) at the start of a line
        pattern = r"(?<=[.!?])[ \t]+|\n{2,}|(?<=\n)\s*[-*•](?= )|(?=^#{1,6}\s)"
        sentence_split = re.split(pattern, text, flags=re.MULTILINE)

        return [
            sentence.replace(placeholder, ".").strip() for sentence in sentence_split
        ]

    def _breakpoints_split(self) -> list[list[str]]:
        """
        Splits documents contents on potential breakpoints.

        :return: documents split on potential breakpoints
        :rtype: list[list[str]]
        """
        breakpoint_chunks = []

        for doc_content in self.docs_contents:
            sentences_split = self._split_sentences(doc_content)
            merged_split = self._merge_too_small_texts(
                sentences_split, self.min_breakpoint_chunk_size
            )
            breakpoint_chunks.append(merged_split)

        return breakpoint_chunks

    @staticmethod
    def _merge_too_small_texts(texts: list[str], size_threshold: int) -> list[str]:
        """
        Concatenates texts of size lower than defined threshold.

        :param texts: list of texts
        :type texts: list[str]

        :param size_threshold: size below which chunks should be merged
        :type size_threshold: int

        :return: list of concatenated texts
        :rtype: list[str]
        """
        merged = []
        temp = []

        for text in texts:
            temp.append(text)
            m = " ".join(temp)
            if len(m) > size_threshold:
                merged.append(m)
                temp = []

        if temp:
            merged.append(" ".join(temp))

        return merged

    def to_dict(self) -> dict[str, Any]:
        """
        Return dictionary that can be used to recreate an instance of the HybridSemanticChunker.

        :return: dictionary which can be used to recreate an instance of the HybridSemanticChunker.
        :rtype: dict
        """
        params = (
            "tfidf_buffer_size",
            "embedding_buffer_size",
            "tfidf_weight",
            "embedding_weight",
            "chunk_size",
        )
        ret = {k: v for k, v in self.__dict__.items() if k in params}
        ret["embeddings"] = self.embeddings.to_dict()

        return ret

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HybridSemanticChunker":
        """
        Create an instance from the dictionary.

        :param d: dictionary that can be used to create an instance of the HybridSemanticChunker.
        :type d: HybridSemanticChunker
        """
        return HybridSemanticChunker(
            embeddings=cast(BaseEmbeddings, BaseEmbeddings.from_dict(d["embeddings"])),
            chunk_size=d["chunk_size"],
            embedding_buffer_size=d["embedding_buffer_size"],
            tfidf_buffer_size=d["tfidf_buffer_size"],
            embedding_weight=d["embedding_weight"],
            tfidf_weight=d["tfidf_weight"],
        )
