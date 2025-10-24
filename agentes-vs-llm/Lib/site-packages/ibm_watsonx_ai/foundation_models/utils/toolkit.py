#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import copy
import json
from typing import Any

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import (
    MissingToolRequiredProperties,
    ResourceByNameNotFound,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource


class Tool(WMLResource):
    """Instantiate the utility agent tool.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param name: name of the tool
    :type name: str

    :param description: description of what the tool is used for
    :type description: str

    :param agent_description: the precise instruction to agent LLMs and should be treated as part of the system prompt, if not provided, `description` can be used in its place
    :type agent_description: str, optional

    :param input_schema: schema of the input that is provided when running the tool if applicable
    :type input_schema: dict, optional

    :param config_schema: schema of the config that is provided when running the tool if applicable
    :type config_schema: dict, optional

    :param config: configuration options that can be passed for some tools, must match the config schema for the tool
    :type config: dict, optional

    """

    def __init__(
        self,
        api_client: APIClient,
        name: str,
        description: str,
        agent_description: str | None = None,
        input_schema: dict | None = None,
        config_schema: dict | None = None,
        config: dict | None = None,
    ):
        self._client = api_client

        Tool._validate_type(name, "name", str)
        Tool._validate_type(input_schema, "input_schema", dict, False)
        Tool._validate_type(config_schema, "config_schema", dict, False)
        Tool._validate_type(config, "config", dict, False)

        self.name = name
        self.description = description
        self.agent_description = agent_description
        self.input_schema = input_schema
        self.config_schema = config_schema
        self.config = config

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

        if self.input_schema is not None:
            self._input_schema_required = self.input_schema.get("required")

    def run(
        self,
        input: str | dict,
        config: dict | None = None,
    ) -> dict:
        """Run a utility agent tool given `input`.

        :param input: input to be used when running tool
        :type input:
            - **str** - if running tool has no `input_schema`
            - **dict** - if running tool has `input_schema`

        :param config: configuration options that can be passed for some tools, must match the config schema for the tool
        :type config: dict, optional

        :return: the output from running the tool
        :rtype: dict

        **Example for the tool without input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search = toolkit.get_tool(tool_name='GoogleSearch')
            result = google_search.run(input="Search IBM")

        **Example for the tool with input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            weather_tool = toolkit.get_tool(tool_name='Weather')
            tool_input = {"location": "New York"}
            result = weather_tool.run(input=tool_input)

        """
        if self.input_schema is None:
            Tool._validate_type(input, "input", str)
        else:
            Tool._validate_type(input, "input", dict)
            if self._input_schema_required and any(
                req not in input for req in self._input_schema_required
            ):
                raise MissingToolRequiredProperties(self._input_schema_required)

        payload = {
            "input": input,
            "tool_name": self.name,
        }

        config = config or self.config

        if config and self.config_schema:
            payload.update({"config": config})  # type: ignore[dict-item]

        response = self._client.httpx_client.post(
            url=self._client._href_definitions.get_utility_agent_tools_run_href(),
            json=payload,
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "run tool", response)

    def __getitem__(self, key: str) -> Any:
        # For backward compatibility in Toolkit.get_tools
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def get(self, key: str, default: Any = None) -> Any:
        # For backward compatibility in Toolkit.get_tools
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __repr__(self) -> str:
        return (
            f'Tool(name="{self.name}", description="{self.description}", '
            f'agent_description="{self.agent_description}", '
            f"input_schema={self.input_schema}, "
            f"config_schema={self.config_schema}, "
            f"config={self.config}, "
            f"api_client={self._client})"
        )


class Toolkit(WMLResource):
    """Toolkit for utility agent tools.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param params: dict of config parameters for each tool, e.g. {"GoogleSearch": {"maxResults": 2}}
    :type params: dict[str, dict], optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.utils import Toolkit

        credentials = Credentials(
            url = "<url>",
            api_key = IAM_API_KEY
        )
        tools_params = {
            "GoogleSearch": {"maxResults": 2}
        }

        api_client = APIClient(credentials)
        toolkit = Toolkit(api_client=api_client, params=tools_params)

    """

    def __init__(self, api_client: APIClient, params: dict[str, dict] | None = None):
        self._client = api_client
        self.params = params
        self._tools: list[Tool] | None = None
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    def get_tools(self) -> list[Tool]:
        """Get list of available utility agent tools. Cache tools as Tool objects on first call in Toolkit instance.

        :return: list of available tools
        :rtype: list[Tool]

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            tools = toolkit.get_tools()

        """
        if self._tools is None:
            response = self._client.httpx_client.get(
                url=self._client._href_definitions.get_utility_agent_tools_href(),
                headers=self._client._get_headers(),
            )
            resources = self._handle_response(
                200, "getting utility agent tools", response
            ).get("resources", [])

            self._tools = [
                Tool(
                    api_client=self._client,
                    name=r["name"],
                    description=r["description"],
                    agent_description=r.get("agent_description"),
                    input_schema=r.get("input_schema"),
                    config_schema=r.get("config_schema"),
                    config=(self.params or {}).get(r["name"]),
                )
                for r in resources
            ]
        return self._tools

    def get_tool(self, tool_name: str) -> Tool:
        """Get a utility agent tool with the given `tool_name`.

        :param tool_name: name of a specific tool
        :type tool_name: str

        :return: tool with a given name
        :rtype: Tool

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search = toolkit.get_tool(tool_name='GoogleSearch')

        """
        Toolkit._validate_type(tool_name, "tool_name", str)

        tools = self.get_tools()

        tool = next(filter(lambda el: el.name == tool_name, tools), None)
        if tool is None:
            raise ResourceByNameNotFound(tool_name, "utility agent tool")
        else:
            return tool


def convert_to_watsonx_tool(utility_tool: Tool) -> dict:
    """Convert utility agent tool to watsonx tool format.

    :param utility_tool: utility agent tool
    :type utility_tool: Tool

    :return: watsonx tool structure
    :rtype: dict

    **Examples**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models.utils import Toolkit

        toolkit = Toolkit(api_client)
        weather_tool = toolkit.get_tool("Weather")
        convert_to_watsonx_tool(weather_tool)

        # Return
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "Weather",
        #         "description": "Find the weather for a city.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "location": {
        #                     "title": "location",
        #                     "description": "Name of the location",
        #                     "type": "string",
        #                 },
        #                 "country": {
        #                     "title": "country",
        #                     "description": "Name of the state or country",
        #                     "type": "string",
        #                 },
        #             },
        #             "required": ["location"],
        #         },
        #     },
        # }

    """

    def parse_parameters(input_schema: dict | None) -> dict:
        if input_schema:
            parameters = copy.deepcopy(input_schema)
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "description": "Input to be used when running tool.",
                        "type": "string",
                    },
                },
                "required": ["input"],
            }

        return parameters

    tool = {
        "type": "function",
        "function": {
            "name": utility_tool.name,
            "description": utility_tool.description,
            "parameters": parse_parameters(utility_tool.input_schema),
        },
    }
    return tool


def convert_to_utility_tool_call(tool_call: dict) -> dict:
    """Convert json format tool call to utility tool call format.

    :param tool_call: watsonx tool call
    :type tool_call: dict

    :return: utility tool call
    :rtype: dict

    **Examples**

    .. code-block:: python

        tool_call = {
            "id": "rcWg61ytv",
            "type": "function",
            "function": {"name": "GoogleSearch", "arguments": '{"input": "IBM"}'},
        }
        convert_to_utility_tool_call(tool_call)

        # Return
        # {"input": "IBM", "tool_name": "GoogleSearch"}

    """
    tool_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]
    try:
        json_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        raise Exception(f"Could not parse {arguments} as json.")
    input_data = json_arguments.get("input", {}) or {
        k: v for k, v in json_arguments.items()
    }

    return {
        "tool_name": tool_name,
        "input": input_data,
    }
