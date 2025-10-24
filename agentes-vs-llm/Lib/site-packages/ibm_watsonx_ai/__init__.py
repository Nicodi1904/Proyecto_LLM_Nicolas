#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from importlib.metadata import version

package_name = __name__.replace("_", "-")
__version__ = version(package_name)

from ibm_watsonx_ai.client import APIClient  # noqa: E402
from ibm_watsonx_ai.credentials import Credentials  # noqa: E402
from ibm_watsonx_ai.utils.enums import AssetDuplicateAction  # noqa: E402

APIClient.version = __version__

__all__ = ["APIClient", "AssetDuplicateAction", "Credentials", "package_name"]
