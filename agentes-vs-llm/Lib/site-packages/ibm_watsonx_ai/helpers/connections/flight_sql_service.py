#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Any, Iterable, Optional

import pandas as pd
from pyarrow import flight

from ibm_watsonx_ai import APIClient

from .flight_service import BaseFlightConnection
from .utils.flight_utils import (
    CallbackSchema,
    HeaderMiddlewareFactory,
    SimplyCallback,
    _flight_retry,
)

logger = logging.getLogger(__name__)


class FlightSQLClient(BaseFlightConnection):
    """FlightSQLClient object unify the work for data reading from different types of data sources,
    including databases. It uses a Flight Service and `pyarrow` library to connect and transfer the data.

    :param connection_id: ID of db connection asset
    :type connection_id: str

    :param api_client: initialized APIClient object.
    :type api_client: APIClient

    :param project_id: ID of project
    :type project_id: str, optional

    :param space_id: ID of space
    :type space_id: str, optional

    :param callback: required for sending messages
    :type callback: StatusCallback, optional

    :param flight_parameters: pure unchanged flight service parameters that need to be passed to the service
    :type flight_parameters: dict, optional

    :param extra_interaction_properties: extra interaction properties passed in flight params
    :type extra_interaction_properties: dict, optional

    :param max_retry_time: maximal time for retrying in seconds (the whole retrying process should take less than max_retry_time)
    :type max_retry_time: int, optional

    """

    def __init__(
        self,
        connection_id: str,
        api_client: APIClient,
        space_id: Optional[str] = None,
        project_id: Optional[str] = None,
        callback: Optional[CallbackSchema] = None,
        flight_parameters: Optional[dict] = None,
        extra_interaction_properties: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_client=api_client, _logger=logger)

        # callback is used in the backend to send status messages
        self.callback = (
            callback if callback is not None else SimplyCallback(logger=self._logger)
        )
        self._max_retry_time = kwargs.get("max_retry_time", 200)

        self.connection_id = connection_id

        self.flight_parameters = (
            flight_parameters if flight_parameters is not None else {}
        )

        if space_id is None and project_id is None:
            error_message = "Either space_id or project_id is required."
            raise ValueError(error_message)

        self._api_client = api_client

        self.additional_connection_args = {}
        if os.environ.get("TLS_ROOT_CERTS_PATH"):
            self.additional_connection_args["tls_root_certs"] = os.environ.get(
                "TLS_ROOT_CERTS_PATH"
            )

        self.extra_interaction_properties = extra_interaction_properties or {}

        # Set flight location and port
        self._set_default_flight_location()

        self._base_command = {}

        if space_id is not None:
            self._base_command["space_id"] = space_id
        else:
            self._base_command["project_id"] = project_id

        self._base_command["asset_id"] = self.connection_id

        self._flight_client = None

    @property
    def flight_client(self) -> flight.FlightClient:
        if self._flight_client is None:
            error_message = (
                "Flight client is not initialized. "
                "Instance of FlightSQLClient should be used as a context manager."
            )
            raise ValueError(error_message)
        return self._flight_client

    def __enter__(self):
        self._flight_client = self._get_flight_client()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._flight_client.wait_for_available(10)
        self._flight_client.close()

    def _get_flight_client(self) -> flight.FlightClient:
        return flight.FlightClient(
            location=f"grpc+tls://{self.flight_location}:{self.flight_port}",
            disable_server_verification=True,
            override_hostname=self.flight_location,
            middleware=[
                HeaderMiddlewareFactory(headers=self._api_client.get_headers())
            ],
            **self.additional_connection_args,
        )

    def _get_source_command(
        self, select_statement: str | None = None, **kwargs: Any
    ) -> str:
        """Get source command for flight service."""
        command = self._base_command.copy()

        if self.flight_parameters is not None:
            command |= self.flight_parameters

        interaction_properties: dict = {}

        if self.extra_interaction_properties:
            interaction_properties |= self.extra_interaction_properties

        if (
            kwargs_interaction_properties := kwargs.pop("interaction_properties", None)
        ) is not None:
            if isinstance(kwargs_interaction_properties, dict):
                interaction_properties |= kwargs_interaction_properties
            else:
                raise TypeError("interaction_properties param should be of type dict")

        if select_statement is not None:
            interaction_properties["select_statement"] = select_statement

        if interaction_properties:
            command["interaction_properties"] = interaction_properties

        for key, value in kwargs.items():
            command[key] = value

        return json.dumps(command)

    @_flight_retry()
    def _get_endpoints(
        self,
        select_statement: str | None = None,
        **kwargs: Any,
    ) -> Iterable[flight.FlightEndpoint]:
        """Listing all available Flight Service endpoints (one endpoint corresponds to one batch)."""
        source_command_kwargs = {}
        if interaction_properties := kwargs.get("interaction_properties"):
            source_command_kwargs["interaction_properties"] = interaction_properties

        source_command = self._get_source_command(
            select_statement=select_statement, **source_command_kwargs
        )

        info = self.flight_client.get_flight_info(
            flight.FlightDescriptor.for_command(source_command)
        )
        return info.endpoints

    @_flight_retry()
    def _execute(self, select_statement: str | None, **kwargs: Any) -> pd.DataFrame:
        """Execute a query on the data source.

        :param query: query to execute
        :type query: str

        :return: query result
        :rtype: pandas.DataFrame
        """
        if unsupported_kwargs := (set(kwargs.keys()) - {"interaction_properties"}):
            error_message = (
                f"Not supported keyword argument(s): {list(unsupported_kwargs)}"
            )
            raise TypeError(error_message)

        def read_thread(
            flight_client: flight.FlightClient, endpoint: flight.FlightEndpoint
        ) -> pd.DataFrame:
            reader = flight_client.do_get(endpoint.ticket)
            return reader.read_pandas()

        endpoints = self._get_endpoints(select_statement=select_statement, **kwargs)

        # Limit max concurrent threads to 10
        with ThreadPoolExecutor(max_workers=10) as executor:
            df_list = list(
                executor.map(
                    read_thread, [self.flight_client] * len(endpoints), endpoints
                )
            )

        return pd.concat(df_list)

    def execute(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """Execute a query on the data source.

        :param query: query to execute
        :type query: str

        :return: query result
        :rtype: pandas.DataFrame
        """
        return self._execute(select_statement=query, **kwargs)

    @_flight_retry()
    def get_tables(self, schema: str) -> dict:
        """Get available tables in the schema.

        :param schema: Schema name
        :type schema: str

        :return: get available tables in schema
        :rtype: dict
        """
        additional_params = {
            "path": f"/{schema}",
            "discovery_filters": {
                "include_system": "false",
                "include_table": "true",
                "include_view": "true",
            },
            "context": "source",
        }

        command = self._get_source_command(**additional_params)
        action = flight.Action("discovery", command.encode("utf-8"))

        action_res = self.flight_client.do_action(action)
        # Retrieve first chunk to read a schema
        first_chunk = dict(json.loads(next(action_res).body.to_pybytes()))
        return reduce(
            lambda left_chunk, right_chunk: FlightSQLClient._reduce_discovery_chunks(
                left_chunk,
                json.loads(right_chunk.body.to_pybytes()),
                reduce_fields=["assets", "total_count"],
            ),
            action_res,
            first_chunk,
        )

    @_flight_retry()
    def get_table_info(self, schema: str, table_name: str, **kwargs: Any) -> dict:
        """Get info about table from given schema."""
        extended_metadata = kwargs.get("extended_metadata", False)
        interaction_properties = kwargs.get("interaction_properties", False)

        fetch = "metadata"
        if extended_metadata:
            fetch += ",extended_metadata"
        if interaction_properties:
            fetch += ",interaction"

        additional_params = {
            "path": f"/{schema}/{table_name}",
            "detail": "true",
            "fetch": fetch,
            "context": "source",
        }

        command = self._get_source_command(**additional_params)
        action = flight.Action("discovery", command.encode("utf-8"))

        action_res = self.flight_client.do_action(action)
        table_info_raw = next(action_res)
        table_info = json.loads(table_info_raw.body.to_pybytes())

        return table_info

    @_flight_retry()
    def get_schemas(self) -> dict:
        """Get available schemas.

        :return: available schemas
        :rtype: dict
        """
        additional_params = {"path": "/", "detail": "true", "context": "source"}

        command = self._get_source_command(**additional_params)
        action = flight.Action("discovery", command.encode("utf-8"))

        action_res = self.flight_client.do_action(action)

        # Retrieve first chunk to read a schema
        first_chunk = dict(json.loads(next(action_res).body.to_pybytes()))
        return reduce(
            lambda left_chunk, right_chunk: FlightSQLClient._reduce_discovery_chunks(
                left_chunk,
                json.loads(right_chunk.body.to_pybytes()),
                reduce_fields=["assets", "totalCount"],
            ),
            action_res,
            first_chunk,
        )

    @staticmethod
    def _reduce_discovery_chunks(
        left_chunk: dict, right_chunk: dict, reduce_fields: list[str]
    ) -> dict:
        for field in reduce_fields:
            if field in right_chunk and field in left_chunk:
                left_chunk[field] += right_chunk[field]

        return left_chunk

    def get_n_first_rows(
        self, schema: str, table_name: str, n: int = 3
    ) -> pd.DataFrame:
        """Get the first n rows of a table.

        :param schema: name of the schema
        :type schema: str

        :param table_name: name of the table
        :type table_name: str

        :param n: number of rows to return, defaults to 3
        :type n: int, optional

        :return: first n rows of the table
        :rtype: pd.DataFrame
        """
        extra_interaction_properties = {
            "schema_name": schema,
            "row_limit": n,
            "table_name": table_name,
        }

        return self._execute(None, interaction_properties=extra_interaction_properties)
