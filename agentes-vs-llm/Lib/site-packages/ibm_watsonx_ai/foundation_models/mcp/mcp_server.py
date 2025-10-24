#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
from inspect import Parameter, signature
from typing import Any, Callable, Union, get_args

from mcp.server.fastmcp import FastMCP

from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit


class MCPServer(FastMCP):
    """MCP (Model Context Protocol) server that uses watsonx.ai Utility Agent Tools.

    :param toolkit: toolkit
    :type toolkit: Toolkit

    :param name: name of a server (FastMCP)
    :type name: str, optional

    :param instructions: instructions to a server (FastMCP)
    :type instructions: str, optional

    :param settings: arbitrary keyword arguments for FastMCP server
    :type settings: Any, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.utils import Toolkit
        from ibm_watsonx_ai.foundation_models.mcp import MCPServer

        credentials = Credentials(
            url = "<url>",
            api_key = IAM_API_KEY
        )
        api_client = APIClient(credentials)
        tools_params = {
            "GoogleSearch": {"maxResults": 2}
            "RAGQuery": {"vectorIndexId": "<vector_index_id>", "projectId": "<project_id>", "spaceId": ""},
        }
        toolkit = Toolkit(api_client=api_client, params=tools_params)
        server = MCPServer(toolkit)

    """

    def __init__(
        self,
        toolkit: Toolkit,
        name: str | None = None,
        instructions: str | None = None,
        **settings: Any,
    ):
        super().__init__(name=name, instructions=instructions, **settings)
        self.toolkit = toolkit

        for tool in self.toolkit.get_tools():
            self._create_tool(tool)

    def _create_tool(self, tool: Tool) -> Any:
        def parse_response(response: dict) -> Any:
            output = response["output"]

            try:
                parsed = json.loads(output)
                return parsed
            except json.JSONDecodeError:
                return output

        if tool.input_schema is None:

            async def fnc(input: str) -> Any:
                res = tool.run(input)
                return parse_response(res)

        else:

            def base_fnc(**kwargs: Any) -> Any:
                res = tool.run(kwargs)
                return parse_response(res)

            schema = self._json_schema_to_type(tool.input_schema)
            fnc = self._modify_argument_types(base_fnc, schema)
        fnc.__name__ = tool.name

        return self.tool(name=tool.name, description=tool.description)(fnc)

    @staticmethod
    def _json_schema_to_type(schema: dict) -> dict:
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        type_mapping: dict[str, type[Any]] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        fields: dict[str, tuple[type[Any], ...]] = {}

        for field_name, field_schema in properties.items():
            field_type = field_schema.get("type", "string")
            py_type = type_mapping.get(field_type, object)

            if field_name in required_fields:
                fields[field_name] = (py_type,)
            else:
                fields[field_name] = (py_type, type(None))

        return fields

    @staticmethod
    def _modify_argument_types(fn: Callable, new_types: dict) -> Callable:
        sig = signature(fn)

        new_params = []

        for param_name, param_type in new_types.items():
            if param_name not in sig.parameters:
                if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
                    union_args = get_args(param_type)
                    if type(None) in union_args:
                        new_param = Parameter(
                            param_name,
                            Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=param_type,
                            default=None,
                        )
                    else:
                        new_param = Parameter(
                            param_name,
                            Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=param_type,
                        )
                else:
                    new_param = Parameter(
                        param_name,
                        Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=param_type,
                    )
                new_params.append(new_param)

        new_sig = sig.replace(parameters=new_params)

        def new_fn(*args: Any, **kwargs: Any) -> Callable:
            bound_args = new_sig.bind(*args, **kwargs)
            updated_args = bound_args.arguments
            unwrapped_kwargs = {k: v for k, v in updated_args.items() if v is not None}
            return fn(**unwrapped_kwargs)

        new_fn.__signature__ = new_sig  # type: ignore[attr-defined]

        return new_fn
