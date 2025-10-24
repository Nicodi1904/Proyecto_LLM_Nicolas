#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["TextLoader"]

import io
import json
import logging
from queue import Empty
from typing import TYPE_CHECKING, Any, Iterator

from ibm_watsonx_ai.helpers.remote_document import RemoteDocument
from ibm_watsonx_ai.utils import DisableWarningsLogger
from ibm_watsonx_ai.wml_client_error import (
    LoadingDocumentError,
    MissingExtension,
    WMLClientError,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from pandas import DataFrame

logger = logging.getLogger(__name__)


def _prepare_iterator(el: Any) -> Iterator[Any]:
    from langchain_core.documents import Document

    if hasattr(el, "__iter__") and not isinstance(el, Document):  # is iterable
        yield from el
    else:
        yield el


def _asynch_download(load_doc):
    """Helper function for parallel downloading documents (full asynchronous version)."""

    def asynch_download(args):
        (q_input, qs_output) = args

        while True:
            try:
                i, doc = q_input.get(block=False)
                try:
                    loaded_doc = load_doc(doc)

                    for el in _prepare_iterator(loaded_doc):
                        qs_output[i].put(el)

                    qs_output[i].put(
                        "End"
                    )  # send signal that no more data will be sent via this queue
                except Exception as e:
                    if "cryptography>=3.1 is required for AES algorithm" in str(e):
                        e = "Encrypted files are not supported. Please decrypt your file and try again."
                    elif "cipher" in str(e) and "not supported" in str(e):
                        e = (
                            "Legacy cryptographic algorithm usage detected. To proceed with their use, "
                            "clear the CRYPTOGRAPHY_OPENSSL_NO_LEGACY environment variable before importing "
                            "ibm_watsonx_ai. Support for these algorithms is being phased out, use at your own risk."
                        )
                    qs_output[i].put(LoadingDocumentError(doc.document_id, e))
            except Empty:
                return

    return asynch_download


class TextLoader:
    """
    TextLoader class for extraction txt, pdf, html, docx and md file from bytearray format.

    :param documents: Documents to extraction from bytearray format
    :type documents: RemoteDocument, list[RemoteDocument]

    """

    def __init__(self, document: RemoteDocument) -> None:
        self.file = document

    def load(self) -> Document:
        """
        Load text from bytearray data.
        """
        try:
            from langchain_core.documents import Document as LCDocument
        except ImportError:
            raise MissingExtension("langchain-core")

        file_content = getattr(self.file, "content", None)
        document_id = getattr(self.file, "document_id", None)
        file_type = self.identify_file_type(document_id)

        file_type_handlers = {
            "text/plain": self._txt_to_string,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._docs_to_string,
            "application/pdf": self._pdf_to_string,
            "text/html": self._html_to_string,
            "text/markdown": self._md_to_string,
            "application/vnd.ms-powerpoint": self._pptx_to_string,
            "application/json": self._json_to_string,
            "application/x-yaml": self._yaml_to_string,
            "application/xml": self._xml_to_string,
            "text/csv": self._csv_to_string,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": self._xlsx_to_string,
            "application/vnd.ms-excel": self._xlsx_to_string,
        }

        try:
            handler = file_type_handlers[file_type]
        except KeyError:
            raise WMLClientError(
                f"Unsupported file type: {file_type}. Supported file types: {list(file_type_handlers)}."
            )

        text = handler(file_content)

        metadata = {
            "document_id": document_id,
        }

        return LCDocument(page_content=text, metadata=metadata)

    @staticmethod
    def identify_file_type(filename: str) -> str:
        """
        Identifying file type by bytearray input data
        """
        filename = filename.lower()
        if filename.endswith(".pdf"):
            return "application/pdf"
        elif filename.endswith(".html"):
            return "text/html"
        elif filename.endswith(".docx"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename.endswith(".txt"):
            return "text/plain"
        elif filename.endswith(".md"):
            return "text/markdown"
        elif filename.endswith(".pptx"):
            return "application/vnd.ms-powerpoint"
        elif filename.endswith(".json"):
            return "application/json"
        elif filename.endswith(".yaml"):
            return "application/x-yaml"
        elif filename.endswith(".xml"):
            return "application/xml"
        elif filename.endswith(".csv"):
            return "text/csv"
        elif filename.endswith(".xlsx"):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.endswith(".xls"):
            return "application/vnd.ms-excel"
        else:
            raise WMLClientError("Cannot identify file type.")

    @staticmethod
    def _txt_to_string(binary_data: bytes) -> str:
        return binary_data.decode("utf-8", errors="ignore")

    @staticmethod
    def _docs_to_string(binary_data: bytes) -> str:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise MissingExtension("python-docx")

        with io.BytesIO(binary_data) as open_docx_file:
            doc = DocxDocument(open_docx_file)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)

    @staticmethod
    def _pdf_to_string(binary_data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise MissingExtension("pypdf")

        with io.BytesIO(binary_data) as open_pdf_file:
            with DisableWarningsLogger():
                reader = PdfReader(open_pdf_file)
            full_text = [page.extract_text() for page in reader.pages]
            return "\n".join(full_text)

    @staticmethod
    def _html_to_string(binary_data: bytes) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise MissingExtension("beautifulsoup4")

        soup = BeautifulSoup(binary_data, "html.parser")
        return soup.get_text()

    @staticmethod
    def _md_to_string(binary_data: bytes) -> str:
        try:
            from markdown import markdown
        except ImportError:
            raise MissingExtension("markdown")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise MissingExtension("beautifulsoup4")

        md = binary_data.decode("utf-8", errors="ignore")
        html = markdown(md)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    @staticmethod
    def _pptx_to_string(binary_data: bytes) -> str:
        try:
            from pptx import Presentation
        except ImportError:
            raise MissingExtension("python-pptx")

        prs = Presentation(io.BytesIO(binary_data))
        result = [
            shape.text
            for slide in prs.slides
            for shape in slide.shapes
            if hasattr(shape, "text")
        ]

        return "\n\n".join(result)

    @classmethod
    def _extract_content_from_py_structure(cls, data: Any) -> list[str]:
        match data:
            case dict():
                return [
                    value
                    for key in data
                    for value in [key]
                    + cls._extract_content_from_py_structure(data[key])
                ]
            case list():
                return [
                    value
                    for el in data
                    for value in cls._extract_content_from_py_structure(el)
                ]
            case str():
                return [data]
            case _:
                return []  # ignore non string leaves

    @classmethod
    def _json_to_string(cls, binary_data: bytes) -> str:
        loaded_json = json.loads(binary_data)
        return "\n\n".join(cls._extract_content_from_py_structure(loaded_json))

    @classmethod
    def _yaml_to_string(cls, binary_data: bytes) -> str:
        import yaml

        docs = yaml.load_all(binary_data, Loader=yaml.Loader)
        return "\n\n\n".join(
            [
                "\n\n".join(cls._extract_content_from_py_structure(loaded_yaml))
                for loaded_yaml in docs
            ]
        )

    @classmethod
    def _xml_to_string(cls, binary_data: bytes) -> str:
        from xml.etree import ElementTree

        def xml_element_parser(elem: ElementTree.Element) -> list[dict | str]:
            result = [dict(elem.attrib), {}]

            children = list(elem)
            if children:
                for child in children:
                    child_tag = child.tag
                    child_data = xml_element_parser(child)

                    if child_tag in result[1]:
                        if isinstance(result[1][child_tag], list):
                            result[1][child_tag].append(child_data)
                        else:
                            result[1][child_tag] = [result[1][child_tag], child_data]
                    else:
                        result[1][child_tag] = child_data

            # Add element text to the result list if exists, at 2nd place after attribute
            if text := (elem.text or "").strip():
                result.insert(1, text)

            return result

        root = ElementTree.fromstring(binary_data)
        root_result = []

        if root.attrib:
            root_result.append(root.attrib)
        if root.text is not None and (text := root.text.strip()):
            root_result.append(text)

        for child in root:
            parsed_child = xml_element_parser(child)
            root_result.append({child.tag: parsed_child})

        result = {root.tag: root_result}

        return "\n\n".join(cls._extract_content_from_py_structure(result))

    @classmethod
    def _df_to_key_value_str(cls, df: "DataFrame") -> str:
        selected_dtypes = df.select_dtypes(
            include=["datetime64", "datetime"]
        )  # for readibility of converted columns
        df[selected_dtypes.columns] = selected_dtypes.astype(str)

        return "\n".join(
            [json.dumps(row)[1:-1] for row in json.loads(df.to_json(orient="records"))]
        )

    @classmethod
    def _csv_to_string(cls, binary_data: bytes) -> str:
        import pandas as pd

        return cls._df_to_key_value_str(
            pd.read_csv(io.BytesIO(binary_data), index_col=[0])
        )

    @classmethod
    def _xlsx_to_string(cls, binary_data: bytes) -> str:
        import pandas as pd

        return cls._df_to_key_value_str(pd.read_excel(binary_data))
