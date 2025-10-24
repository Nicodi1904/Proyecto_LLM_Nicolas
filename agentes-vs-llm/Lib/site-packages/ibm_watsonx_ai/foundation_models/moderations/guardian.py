#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ibm_watsonx_ai.foundation_models.schema import (
    BaseSchema,
    GuardianDetectors,
)
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class Guardian(WMLResource):
    """
    Guardian is responsible for text detection using configured detectors.

    :param api_client: The APIClient instance
    :type api_client: APIClient

    :param detectors: A dict of detector configurations
    :type detectors: dict, GuardianDetectors

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.moderations import Guardian

        credentials = Credentials(
            url = "<url>",
            api_key = IAM_API_KEY
        )
        api_client = APIClient(credentials)

        detectors = {
            "granite_guardian": {"threshold": 0.4}
        }

        guardian = Guardian(
            api_client=api_client,
            detectors=detectors
        )

    """

    def __init__(
        self, api_client: APIClient, detectors: dict | GuardianDetectors
    ) -> None:
        self._client = api_client

        Guardian._validate_type(
            detectors, "detectors", [dict, GuardianDetectors], False, True
        )

        self.detectors = detectors

        if isinstance(self.detectors, BaseSchema):
            self.detectors = self.detectors.to_dict()

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    def detect(
        self, text: str, detectors: dict | GuardianDetectors | None = None
    ) -> dict:
        """
        Detects elements in the given text using specified detectors.

        :param text: The input text to analyze
        :type text: str

        :param detectors: A dict of detector configurations
        :type detectors: dict, GuardianDetectors, optional

        **Example:**

        .. code-block:: python

            text = "I would like to say some `Indecent words`."

            response = guardian.detect(text=text)

        """
        Guardian._validate_type(text, "text", str)

        payload: dict = {
            "input": text,
        }

        if detectors is None:
            detectors = deepcopy(self.detectors)

        if isinstance(detectors, BaseSchema):
            detectors = detectors.to_dict()

        payload["detectors"] = detectors

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_text_detection_href(),
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "text detection", response)
