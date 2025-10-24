#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials


class AudioModelInference(WMLResource):
    """
    Instantiate the audio model interface

    :param model: type of model to use
    :type model: str, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import AudioModelInference

        audio_model = AudioModelInference(
            model="<AUDIO MODEL>",
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"
            ),
            project_id=project_id
        )

    """

    def __init__(
        self,
        model: str,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        self.model = model

        if credentials:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials, verify=verify)
        elif api_client:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id:
            self._client.set.default_space(space_id)
        elif project_id:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        WMLResource.__init__(self, __name__, self._client)

    def transcribe(
        self,
        file_path: str | Path,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe audio into text.

        :param file_path: The path to the audio file to transcribe
        :type file_path: str, Path, required

        :param language: Target language to which to transcribe, e.g. 'fr' for French. Default is English.
        :type language: str, optional

        **Example:**

        .. code-block:: python

            file_path = "sample_audio.mp3"

            response = audio_model.transcribe(file_path=file_path)

        """
        self._client._check_if_either_is_set()

        if isinstance(file_path, str):
            file_path = Path(file_path)

        headers = self._client._get_headers()
        headers.pop("Content-Type", None)

        if self._client.default_space_id:
            headers["Authorization"] += f";space_id={self._client.default_space_id}"

        elif self._client.default_project_id:
            headers["Authorization"] += f";project_id={self._client.default_project_id}"

        payload: dict = {"model": self.model}
        if language is not None:
            payload["language"] = language

        with file_path.open("rb") as file:
            files = {"file": (file_path.name, file, "multipart/form-data")}
            response = self._client.httpx_client.post(
                url=self._client._href_definitions.get_audio_transcriptions_href(),
                data=payload,
                files=files,
                headers=headers,
            )

        return self._handle_response(200, "transcribe", response)
