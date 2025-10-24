#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.utils.utils import StrEnum


class TextExtractionsV2ResultFormats(StrEnum):
    ASSEMBLY_JSON = "assembly"
    HTML = "html"
    MARKDOWN = "md"
    PLAIN_TEXT = "plain_text"
    PAGE_IMAGES = "page_images"
    TABLES_JSON = "tables_json"
