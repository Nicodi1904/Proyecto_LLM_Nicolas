#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import json
import logging
import mimetypes
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from subprocess import check_call
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Iterable,
    Type,
    TypeAlias,
    cast,
)
from warnings import warn

import httpx
import numpy
import requests as _requests
from anyio import AsyncFile
from packaging import version

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai import __version__ as package_version
from ibm_watsonx_ai import package_name
from ibm_watsonx_ai._wrappers.requests import HTTPX_DEFAULT_LIMIT, HTTPX_DEFAULT_TIMEOUT
from ibm_watsonx_ai.href_definitions import HrefDefinitions
from ibm_watsonx_ai.wml_client_error import (
    CannotInstallLibrary,
    GovCloudEnvironmentConsentError,
    WMLClientError,
)

if TYPE_CHECKING:
    import collections
    from types import TracebackType

    import pyspark
    from IPython.display import HTML

    from ibm_watsonx_ai import APIClient

    PipelineType: TypeAlias = Any
    MLModelType: TypeAlias = Any

INSTANCE_DETAILS_TYPE = "instance_details_type"
PIPELINE_DETAILS_TYPE = "pipeline_details_type"
DEPLOYMENT_DETAILS_TYPE = "deployment_details_type"
EXPERIMENT_RUN_DETAILS_TYPE = "experiment_run_details_type"
MODEL_DETAILS_TYPE = "model_details_type"
DEFINITION_DETAILS_TYPE = "definition_details_type"
EXPERIMENT_DETAILS_TYPE = "experiment_details_type"
TRAINING_RUN_DETAILS_TYPE = "training_run_details_type"
FUNCTION_DETAILS_TYPE = "function_details_type"
DATA_ASSETS_DETAILS_TYPE = "data_assets_details_type"
SW_SPEC_DETAILS_TYPE = "sw_spec_details_type"
HW_SPEC_DETAILS_TYPE = "hw_spec_details_type"
RUNTIME_SPEC_DETAILS_TYPE = "runtime_spec_details_type"
LIBRARY_DETAILS_TYPE = "library_details_type"
SPACES_DETAILS_TYPE = "spaces_details_type"
MEMBER_DETAILS_TYPE = "member_details_type"
CONNECTION_DETAILS_TYPE = "connection_details_type"
PKG_EXTN_DETAILS_TYPE = "pkg_extn_details_type"
UNKNOWN_ARRAY_TYPE = "resource_type"
UNKNOWN_TYPE = "unknown_type"
SPACES_IMPORTS_DETAILS_TYPE = "spaces_imports_details_type"
SPACES_EXPORTS_DETAILS_TYPE = "spaces_exports_details_type"

SPARK_MLLIB = "mllib"
SPSS_FRAMEWORK = "spss-modeler"
TENSORFLOW_FRAMEWORK = "tensorflow"
XGBOOST_FRAMEWORK = "xgboost"
SCIKIT_LEARN_FRAMEWORK = "scikit-learn"
PMML_FRAMEWORK = "pmml"


def _get_id_from_deprecated_uid(
    kwargs: dict, resource_id: str | None, resource_name: str, can_be_none: bool = False
) -> str:
    if (resource_uid := kwargs.get(resource_name + "_uid")) is not None:
        parameter_deprecated_warning = (
            f"`{resource_name}_uid` parameter is deprecated, "
            f"please use `{resource_name}_id` instead"
        )
        warn(parameter_deprecated_warning, category=DeprecationWarning)
        if not resource_id:
            resource_id = resource_uid
    elif not can_be_none and resource_uid is None and resource_id is None:
        raise TypeError(
            f"Function missing 1 required positional argument: '{resource_name}_id'"
        )

    return resource_id


@dataclass
class HttpClientConfig:
    """
    A class for storing parameters used to initialize an `httpx.Client`.
    Using this class is recommended when adjusting timeouts or limits instead of providing a separate `httpx.Client`.
    :param timeout: The timeout configuration for sending requests.
    :type timeout: httpx.Timeout, optional

    :param limits: The limits configuration to control the connection pool size.
    :type limits: httpx.Limits, optional
    """

    timeout: float | httpx.Timeout = field(
        default_factory=lambda: HTTPX_DEFAULT_TIMEOUT
    )
    limits: httpx.Limits = field(default_factory=lambda: HTTPX_DEFAULT_LIMIT)


DEFAULT_HTTP_CLIENT_CONFIG = HttpClientConfig(
    timeout=HTTPX_DEFAULT_TIMEOUT, limits=HTTPX_DEFAULT_LIMIT
)


def print_text_header_h1(title: str) -> None:
    print("\n\n" + ("#" * len(title)) + "\n")
    print(title)
    print("\n" + ("#" * len(title)) + "\n\n")


def print_text_header_h2(title: str) -> None:
    print("\n\n" + ("-" * len(title)))
    print(title)
    print(("-" * len(title)) + "\n\n")


def get_type_of_details(details: dict) -> str:
    if "resources" in details:
        return UNKNOWN_ARRAY_TYPE
    elif details is None:
        raise WMLClientError("Details doesn't exist.")
    else:
        try:
            plan = "plan" in details["entity"]

            if plan:
                return INSTANCE_DETAILS_TYPE

            if (
                re.search(r"\/wml_instances\/[^\/]+$", details["metadata"]["url"])
                is not None
            ):
                return INSTANCE_DETAILS_TYPE
        except Exception:
            pass
        try:
            if (
                re.search(r"\/pipelines\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return PIPELINE_DETAILS_TYPE
        except Exception:
            pass
        try:
            if (
                "href" in details["metadata"]
                and re.search(r"\/deployments\/[^\/]+$", details["metadata"]["href"])
                is not None
                or re.search(r"\/deployments\/[^\/]+$", details["metadata"]["id"])
                is not None
                or "virtual_deployment_downloads" in details["entity"]["status"]
            ):
                return DEPLOYMENT_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/experiments\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return EXPERIMENT_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/trainings\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return TRAINING_RUN_DETAILS_TYPE
        except Exception:
            pass

        try:
            if re.search(r"\/models\/[^\/]+$", details["metadata"]["href"]) is not None:
                return MODEL_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/functions\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return FUNCTION_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/runtimes\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return RUNTIME_SPEC_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/libraries\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return LIBRARY_DETAILS_TYPE
        except Exception:
            pass

        try:
            if re.search(r"\/spaces\/[^\/]+$", details["metadata"]["href"]) is not None:
                return SPACES_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/members\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return MEMBER_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/members\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return MEMBER_DETAILS_TYPE
        except Exception:
            pass

        try:
            if re.search(r"\/assets\/[^\/]+$", details["metadata"]["href"]) is not None:
                return DATA_ASSETS_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(
                    r"\/software_specifications\/[^\/]+$", details["metadata"]["href"]
                )
                is not None
            ):
                return SW_SPEC_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(
                    r"\/hardware_specifications\/[^\/]+$", details["metadata"]["href"]
                )
                is not None
            ):
                return HW_SPEC_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(
                    r"\/package_extension\/[^\/]+$",
                    details["entity"]["package_extension"]["href"],
                )
                is not None
            ):
                return PKG_EXTN_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/imports\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return SPACES_IMPORTS_DETAILS_TYPE
        except Exception:
            pass

        try:
            if (
                re.search(r"\/exports\/[^\/]+$", details["metadata"]["href"])
                is not None
            ):
                return SPACES_EXPORTS_DETAILS_TYPE
        except Exception:
            pass

        return UNKNOWN_TYPE


def load_model_from_directory(
    framework: dict, directory_path: str
) -> pyspark.ml.pipeline.PipelineModel | None:
    if "mllib" in framework:
        from pyspark.ml import PipelineModel

        return PipelineModel.read().load(directory_path)
    if "spss" in framework:
        pass
    if "tensorflow" in framework:
        pass
    if "scikit" in framework or "xgboost" in framework:
        try:
            try:
                from sklearn.externals import joblib
            except ImportError:
                import joblib
            pkl_files = [x for x in os.listdir(directory_path) if x.endswith(".pkl")]

            if len(pkl_files) < 1:
                raise WMLClientError("No pkl files in directory.")

            model_id = pkl_files[0]
            return joblib.load(os.path.join(directory_path, model_id))
        except Exception as e:
            raise WMLClientError("Cannot load model from pkl file.", e)
    if "pmml" in framework:
        return None
    else:
        raise WMLClientError("Invalid framework specified: '{}'.".format(framework))


def save_model_to_file(
    model: MLModelType, framework: str, base_path: str, filename: str
) -> None:
    if filename.find(".") != -1:
        base_name = filename[: filename.find(".") + 1]
    else:
        base_name = filename

    if framework == SPARK_MLLIB:
        model.write.overwrite.save(os.path.join(base_path, base_name))
    elif framework == SPSS_FRAMEWORK:
        pass
    elif framework == TENSORFLOW_FRAMEWORK:
        pass
    elif framework == XGBOOST_FRAMEWORK:
        pass
    elif framework == SCIKIT_LEARN_FRAMEWORK:
        os.makedirs(os.path.join(base_path, base_name))
        try:
            from sklearn.externals import joblib
        except ImportError:
            import joblib
        joblib.dump(model, os.path.join(base_path, base_name, base_name + ".pkl"))
    elif framework == PMML_FRAMEWORK:
        pass
    else:
        raise WMLClientError("Invalid framework specified: '{}'.".format(framework))


def format_metrics(latest_metrics_list: list[dict]) -> str:
    formatted_metrics = ""

    for i in latest_metrics_list:
        values = i["values"]

        if len(values) > 0:
            sorted_values = sorted(values, key=lambda k: k["name"])
        else:
            sorted_values = values

        for j in sorted_values:
            formatted_metrics = (
                formatted_metrics
                + i["phase"]
                + ":"
                + j["name"]
                + "="
                + "{0:.4f}".format(j["value"])
                + "\n"
            )

    return formatted_metrics


def inherited_docstring(
    f: Callable, mapping: dict | None = None, actual_type_override: str | None = None
) -> Callable:
    def dec(obj: Callable) -> Callable:
        if obj.__doc__ or not f.__doc__:
            return obj

        possible_types = {
            "model": "model",
            "function": "function",
            "space": "space",
            "pipeline": "pipeline",
            "experiment": "experiment",
            "member": "space",
            "ai_service": "ai_service",
        }

        available_metanames = {
            "model": "ModelMetaNames",
            "experiment": "ExperimentMetaNames",
            "function": "FunctionMetaNames",
            "pipeline": "PipelineMetaNames",
            "ai_service": "AIServiceMetaNames",
        }

        actual_type = actual_type_override
        for possible, actual in possible_types.items():
            if possible in obj.__name__:
                actual_type = actual

        docs = f.__doc__

        if actual_type:
            docs = docs.replace(
                f"client.{actual_type}s.{f.__name__}",
                "client.repository." + obj.__name__,
            )
            docs = docs.replace(
                f"client._{actual_type}s.{f.__name__}",
                "client.repository." + obj.__name__,
            )

            if actual_type in available_metanames:
                repository_meta_names = available_metanames[actual_type]
                docs = docs.replace(
                    f"_{actual_type}s.ConfigurationMetaNames",
                    f"repository.{repository_meta_names}",
                )
                docs = docs.replace(
                    f"{actual_type}s.ConfigurationMetaNames",
                    f"repository.{repository_meta_names}",
                )
                docs = docs.replace("ConfigurationMetaNames", repository_meta_names)

            if mapping:
                for k in mapping:
                    docs = docs.replace(k, mapping[k])

        obj.__doc__ = docs
        return obj

    return dec


def group_metrics(metrics: list[dict]) -> list | collections.defaultdict:
    grouped_metrics: list | collections.defaultdict = []

    if len(metrics) > 0:
        import collections

        grouped_metrics = collections.defaultdict(list)
        for d in metrics:
            k = d["phase"]
            grouped_metrics[k].append(d)

    return grouped_metrics


class StatusLogger:
    def __init__(self, initial_state: str):
        self.last_state = initial_state
        print(initial_state, end="")

    def log_state(self, state: str) -> None:
        if state == self.last_state:
            print(".", end="")
        else:
            print("\n{}".format(state), end="")
            self.last_state = state

    def __enter__(self) -> StatusLogger:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


def get_file_from_cos(cos_credentials: dict) -> str:
    import ibm_boto3
    from ibm_botocore.client import Config

    client_cos = ibm_boto3.client(
        service_name="s3",
        ibm_api_key_id=cos_credentials["IBM_API_KEY_ID"],
        ibm_auth_endpoint=cos_credentials["IBM_AUTH_ENDPOINT"],
        config=Config(signature_version="oauth"),
        endpoint_url=cos_credentials["ENDPOINT"],
    )

    streaming_body = client_cos.get_object(
        Bucket=cos_credentials["BUCKET"], Key=cos_credentials["FILE"]
    )["Body"]
    training_definition_bytes = streaming_body.read()
    streaming_body.close()
    filename = cos_credentials["FILE"]
    f = open(filename, "wb")
    f.write(training_definition_bytes)
    f.close()

    return filename


def extract_model_from_repository(
    model_id: str, client: APIClient, **kwargs: Any
) -> str:
    """Download and extract archived model from wml repository.

    :param model_id: ID of model
    :type model_id: str
    :param client: client instance
    :type client: APIClient

    :return: extracted directory path
    :rtype: str
    """
    model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
    create_empty_directory(model_id)
    current_dir = os.getcwd()

    os.chdir(model_id)
    model_dir = os.getcwd()

    fname = "downloaded_" + model_id + ".tar.gz"
    client.repository.download(model_id, filename=fname)

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
    else:
        raise WMLClientError("Invalid type. Expected tar.gz")

    os.chdir(current_dir)
    return model_dir


def extract_mlmodel_from_archive(
    archive_path: str, model_id: str, **kwargs: Any
) -> str:
    """Extract archived model under model id directory.

    :param model_id: ID of model
    :type model_id: str
    :param archive_path: path to archived model
    :type archive_path: str

    :return: extracted directory path
    :rtype: str
    """
    model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
    create_empty_directory(model_id)
    current_dir = os.getcwd()

    os.rename(archive_path, os.path.join(model_id, archive_path))

    os.chdir(model_id)

    if archive_path.endswith("tar.gz"):
        tar = tarfile.open(archive_path)
        tar.extractall()
        tar.close()
    else:
        raise WMLClientError("Invalid type. Expected tar.gz")

    os.chdir(current_dir)
    return os.path.join(model_id, "model.mlmodel")


def get_model_filename(directory: str, model_extension: str) -> str:
    logger = logging.getLogger(__name__)
    model_filepath = None

    for file in os.listdir(directory):
        if file.endswith(model_extension):
            if model_filepath is None:
                model_filepath = os.path.join(directory, file)
            else:
                logger.warning(
                    "More than one file with extension '{}'.".format(model_extension)
                )

    if model_filepath is None:
        raise WMLClientError("No file with extension '{}'.".format(model_extension))

    return model_filepath


def delete_directory(directory: str) -> None:
    if os.path.exists(directory):
        shutil.rmtree(directory)


def create_empty_directory(directory: str) -> None:
    delete_directory(directory)
    os.makedirs(directory)


def install_package(package: str) -> None:
    import importlib

    try:
        importlib.import_module(package)
    except ImportError:
        import pip

        pip.main(["install", package])


def is_ipython() -> bool:
    # checks if the code is run in the notebook
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: F821
        return True
    except Exception:
        return False


def create_download_link(file_path: str, title: str = "Download file.") -> HTML | None:
    # creates download link for binary files on notebook filesystem (Watson Studio)

    if is_ipython():
        import base64

        from IPython.display import HTML

        filename = os.path.basename(file_path)

        with open(file_path, "rb") as file:
            b_model = file.read()
        b64 = base64.b64encode(b_model)
        payload = b64.decode()
        html = '<a download="{file_path}" href="data:binary;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload, title=title, file_path=filename)

        return HTML(html)

    return None


def convert_metadata_to_parameters(meta_data: dict) -> list:
    parameters = []

    if meta_data is not None:
        for key, value in meta_data.items():
            parameters.append({"name": str(key), "value": value})

    return parameters


def is_of_python_basic_type(el: object | list | None) -> bool:
    if el is None:
        return True
    elif type(el) in [int, float, bool, str]:
        return True
    elif type(el) in [list, tuple]:
        return all([is_of_python_basic_type(t) for t in cast(Iterable, el)])
    elif type(el) is dict:
        if not all(type(k) is str for k in el.keys()):
            return False

        return is_of_python_basic_type(list(el.values()))
    else:
        return False


def _handle_next_details_response(
    response: httpx.Response | _requests.Response,
    _all: bool,
    _filter_func: Callable | None,
    _silent_response_logging: bool,
) -> tuple[str | None, dict[str, Any]]:
    # Import needs to be inside of function body,
    # because WMLResource imports utils
    from ibm_watsonx_ai.wml_resource import (  # pylint: disable=import-outside-toplevel
        WMLResource,
    )

    details_json = WMLResource._handle_response(
        200,
        "Get next details",
        response,
        _silent_response_logging=_silent_response_logging,
    )

    next_href = details_json.get("next", {"href": None})["href"] if _all else None

    if "resources" in details_json:
        resources = details_json["resources"]
        if not resources:
            next_href = None
    elif "metadata" in details_json:
        resources = [details_json]
    else:
        resources = details_json.get("results", [])

    return next_href, {
        "resources": (_filter_func(resources) if _filter_func else resources)
    }


def next_resource_generator(
    client: APIClient,
    url: str,
    href: str,
    params: dict | None = None,
    _all: bool = False,
    _filter_func: Callable | None = None,
    use_httpx: bool = False,
    _silent_response_logging: bool = False,
) -> Generator[dict, None, None]:
    """
    Generator to produce next list of resources from REST API.

    :param client: Client Instance
    :type client: APIClient

    :param url: URL to the resource
    :type url: str

    :param href: href to the resource
    :type href: str

    :param params: parameters of request
    :type params: dict

    :param _all: if `True`, it will get all entries in 'limited' chunks
    :type _all: bool, optional

    :param _filter_func: filtering function
    :type _filter_func: function, optional

    :param use_httpx: use httpx.Client
    :type use_httpx: bool, optional
    """
    next_href: str | None = href

    request_handler = client.httpx_client if use_httpx else requests

    while next_href is not None:
        if "http" not in next_href:
            next_href = f"{url}/{next_href}"

        response = request_handler.get(
            url=next_href,
            headers=client._get_headers(),
            params=(params if params is not None else client._params()),
        )

        next_href, resources = _handle_next_details_response(
            response, _all, _filter_func, _silent_response_logging
        )

        yield resources


async def anext_resource_generator(
    client: APIClient,
    url: str,
    href: str,
    params: dict | None = None,
    _all: bool = False,
    _filter_func: Callable | None = None,
    _silent_response_logging: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    Generator to produce next list of resources from REST API asynchronously.

    :param client: api client Instance
    :type client: APIClient

    :param url: url to the resource
    :type url: str

    :param href: href to the resource
    :type href: str

    :param params: parameters of request
    :type params: dict

    :param _all: if `True`, it will get all entries in 'limited' chunks
    :type _all: bool, optional

    :param _filter_func: filtering function
    :type _filter_func: function, optional
    """
    next_href: str | None = href

    while next_href is not None:
        if "http" not in next_href:
            next_href = f"{url}/{next_href}"

        response = await client.async_httpx_client.get(
            url=next_href,
            headers=await client._aget_headers(),
            params=(params if params is not None else client._params()),
        )

        next_href, resources = _handle_next_details_response(
            response, _all, _filter_func, _silent_response_logging
        )

        yield resources


class DisableWarningsLogger:
    """Class which disables logging warnings (for example for silent handling WMLClientErrors in try except).

    **Example:**

    .. code-block:: python

        try:
            with DisableWarningsLogger():
                throw_wml_error()
        except WMLClientError:
            success = False

    """

    def __enter__(self) -> None:
        logging.disable(logging.WARNING)

    def __exit__(
        self,
        exit_type: Type[BaseException] | None,
        exit_value: BaseException | None,
        exit_traceback: TracebackType | None,
    ) -> None:
        logging.disable(logging.NOTSET)


def is_lib_installed(
    lib_name: str,
    minimum_version: str | None = None,
    install: bool = False,
) -> bool:
    """Check if provided library is installed on user environment. If not, tries to install it.

    :param lib_name: library name to check
    :type lib_name: str

    :param minimum_version: minimum version of library to check, default: None - check if library is installed in overall
    :type minimum_version: str, optional

    :param install: indicates to install missing or to low version library
    :type install: bool, optional

    :return: information if library is installed: `True` is library is installed, `False` otherwise
    :rtype: bool
    """
    if lib_name in sys.modules:
        installed = True

    elif importlib.util.find_spec(lib_name) is not None:
        installed = True

    else:
        installed = False

    if installed:
        installed_module_version = get_module_version(lib_name)

        if minimum_version is not None:
            if version.parse(installed_module_version) < version.parse(minimum_version):
                if install:
                    install_library(
                        lib_name=lib_name, version=minimum_version, strict=False
                    )

    else:
        if install:
            install_library(lib_name=lib_name, version=minimum_version, strict=False)
            installed = True

    return installed


def install_library(
    lib_name: str, version: str | None = None, strict: bool = False
) -> None:
    """Try to install library.

    :param lib_name: library name to install
    :type lib_name: str

    :param version: version of the library to install
    :type version: str, optional

    :param strict: indicates if we want to install specific version or higher version if available
    :type strict: bool, optional
    """
    try:
        if version is not None:
            check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    f"{lib_name}{'==' if strict else '>='}{version}",
                ]
            )

        else:
            check_call([sys.executable, "-m", "pip", "install", lib_name])

    except Exception as e:
        raise CannotInstallLibrary(lib_name=lib_name, reason=str(e))


def get_module_version(lib_name: str) -> str:
    """Use only when you need to check package version by package name with pip."""
    from importlib.metadata import version

    return version(lib_name)


def prepare_interaction_props_for_cos(source_params: dict, file_name: str) -> dict:
    """If user specified properties for dataset as sheet_name, delimiter etc. we need to
    pass them as interaction properties for Flight Service.

    :param source_params: data source parameters describe data (eg. excel_sheet, encoding etc.)
    :type source_params: dict

    :param file_name: name of the file to download, should consist of file extension
    :type file_name: str

    :return: COS interaction properties for Flight Service
    :rtype: dict
    """
    interaction_properties = {}
    file_format = None

    encoding = source_params.get("encoding", None)

    if ".xls" in file_name or ".xlsx" in file_name:
        file_format = "excel"
        if source_params.get("excel_sheet"):
            interaction_properties["sheet_name"] = str(source_params.get("excel_sheet"))

    elif ".csv" in file_name:
        if encoding is not None:
            interaction_properties["encoding"] = encoding

        input_file_separator = source_params.get("input_file_separator", ",")
        if input_file_separator == ",":
            file_format = "csv"
        else:
            file_format = "delimited"
            interaction_properties["field_delimiter"] = input_file_separator

            if quote_character := source_params.get("quote_character"):
                interaction_properties["quote_character"] = str(quote_character)

    elif ".parquet" in file_name or ".prq" in file_name:
        file_format = "parquet"

    if file_format is not None:
        interaction_properties["file_format"] = file_format

    return interaction_properties


def modify_details_for_script_and_shiny(details_from_get: dict) -> dict:
    """Add the href and id of and asset to the same position as it is returned from the POST method
    it allows the `get_id`/`get_href` method to work with details returned by GET method.

    :param details_from_get: details of script/shiny app acquired using GET method
    :type details_from_get: dict

    :return: details with 'guid' and 'href' key added to 'metadata'
    :rtype: dict
    """
    try:
        details_from_get["metadata"]["href"] = details_from_get["href"]
        details_from_get["metadata"]["guid"] = details_from_get["metadata"]["asset_id"]
    except KeyError:
        pass

    return details_from_get


def is_lale_pipeline(pipeline: PipelineType) -> bool:
    return (
        type(pipeline).__module__ == "lale.operators"
        and type(pipeline).__qualname__ == "TrainedPipeline"
    )


class NumpyTypeEncoder(json.JSONEncoder):
    """Extended json.JSONEncoder to encode correctly numpy types."""

    def default(
        self, obj: numpy.integer | numpy.bool_ | numpy.floating | numpy.ndarray
    ) -> int | bool | float | list | None:
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, numpy.floating):
            return None if numpy.isnan(obj) else float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def _requests_convert_json_to_data(
    data_arg: dict | None, json_arg: dict | None, kwargs_arg: dict
) -> tuple[dict | str | None, dict | None, dict]:
    data = None
    if (js := json_arg) is not None and data_arg is None:
        data = json.dumps(js, cls=NumpyTypeEncoder)

        if kwargs_arg.get("headers") and not get_from_json(
            kwargs_arg, ["headers", "Content-Type"]
        ):
            kwargs_arg["headers"]["Content-Type"] = "application/json"
    return (
        (data, json_arg, kwargs_arg)
        if data_arg is None and js is not None
        else (data_arg, json_arg, kwargs_arg)
    )


def get_user_agent_header() -> str:
    """
    Function which return User-Agent header
    """
    lang = "python"
    try:
        arch = os.uname().machine
    except Exception:
        arch = None

    try:
        operation_system = platform.system().lower()
    except Exception:
        operation_system = None

    try:
        python_version = platform.python_version()
    except Exception:
        python_version = None

    return f"{package_name}/{package_version} (lang={lang}; arch={arch}; os={operation_system}; python.version={python_version})"


def _get_expiration_datetime_from_headers(headers: dict) -> datetime | None:
    try:
        from ibm_watsonx_ai.utils.auth.base_auth import _get_token_info

        token = headers.get("Authorization", " ").split(" ")[-1]

        token_info = _get_token_info(token)

        token_expire = token_info.get("exp")

        return datetime.fromtimestamp(token_expire)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Token retrieval failed, error: {e}")
        return


class StrEnum(str, Enum):
    """
    External class created for the needs of auto-generated enums

    UseCase of StrEnum:
    When we call print function on StrEnum attribute we are getting value ot them instead of Enum object

    Example of StrEnum
    TestEnum.Enum1 == "enum1" --> True

    Example of Enum
    TestEnum.Enum1 == "enum1 --> False
    """

    def __str__(self) -> str:
        return self.value

    @classmethod
    def show(cls) -> None:
        elements_dict = {element.name: element.value for element in cls}
        print(elements_dict)


class _APIClientSession(requests.Session):
    """Wrapper for requests.Session to use in APIClient initialization."""

    def __init__(self, client: APIClient):
        super().__init__()
        self._client = client

    def __exit__(self, *args):
        super().__exit__(*args)
        # when APIClient initialization is over we use requests module instead of requests.Session
        self._client._session = requests


def _get_default_args(func: Callable) -> dict[str, Any]:
    """Get a mapping that stores func parameters as keys and their default values as values.

    :param func: The function that is checked
    :type func: Callable

    :return: Dictionary representing function parameters and corresponding default values.
    :rtype: dict[str, Any]
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _create_href_definitions(client: APIClient) -> HrefDefinitions:
    return HrefDefinitions(
        url=client.credentials.url,
        instance_id=client.credentials.instance_id,
        version=client.credentials.version,
        bedrock_url=client.credentials.bedrock_url,
        cloud_platform_spaces=client.CLOUD_PLATFORM_SPACES,
        cp4d_platform_spaces=client.ICP_PLATFORM_SPACES,
        platform_url=client.PLATFORM_URL,
        project_type=client.project_type,
        _use_fm_ga_api=client._use_fm_ga_api,
    )


def _handle_fl_removal(client: APIClient):
    if client.ICP_PLATFORM_SPACES and client.CPD_version >= 5.2:
        raise WMLClientError(
            "Federated Learning is removed in IBM Cloud Pak for Data 5.2 and above."
        )
    else:
        FL_DEPRECATED_WARNING = "Federated Learning is deprecated and will be removed in IBM Cloud Pak for Data 5.2."

        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)


def _requests_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Iterable[int] = (500, 502, 503, 504, 520, 521, 524),
    session: requests.Session | None = None,
) -> requests.Session:
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def raise_exception_about_unsupported_on_cloud(func: Callable) -> Callable:
    from ibm_watsonx_ai.wml_resource import WMLResource

    @functools.wraps(func)
    def wrapper(resource: WMLResource, *args: Any, **kwargs: Any) -> Any:
        if resource._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(
                error_msg=f"{resource.__class__} is not supported on IBM watsonx.ai for IBM Cloud!"
            )
        return func(resource, *args, **kwargs)

    return wrapper


def content_type_for(filepath: str, default: str = "application/octet-stream") -> str:
    """
    Return the best‐guess Content-Type for a file path, falling back to `default` if unknown.
    """
    # 1) Make sure .yaml/.yml map to text/yaml
    mimetypes.add_type("text/yaml", ".yaml")
    mimetypes.add_type("text/yaml", ".yml")

    ext = Path(filepath).suffix
    mime = mimetypes.types_map.get(ext.lower())
    return mime or default


def get_filename_from_asset_details(asset_details: dict) -> str | None:
    """Return filename from asset details."""
    filename = asset_details["metadata"].get("resource_key") or asset_details[
        "metadata"
    ].get("attachment_name")
    return filename.split("/")[-1] if filename else None


GOV_CLOUD_CONSENT_FORMULA = """
You are accessing a U.S. Government (USG) Information System (IS) that is provided for USG-authorized use only.
By using this IS (which includes any device attached to this IS), you consent to the following conditions:
 - All actions on this system are tracked and recorded
 - Unauthorized use of this system is prohibited and is subject to criminal and civil penalties
 - You are prohibited from export, copy, screenshot or print any data.
 - By using this system you consent to monitoring and recording
 """


def _validate_gov_cloud_env(url: str, logger: logging.Logger) -> None:
    """Validate GovCloud environment.

    :param url: URL
    :type url: str
    :param logger: logger instance
    :type logger: logging.Logger

    :raises GovCloudEnvironmentConsentError: when url is GovCloud type but env var
                                            `WATSONX_ACCEPT_GOV_ENV` is not set to "True"
    """
    parsed_url = urllib.parse.urlparse(url)
    hostname = (parsed_url.hostname or "").lower()
    if hostname.endswith("ibmforusgov.com"):
        gov_env_var_name = "WATSONX_ACCEPT_GOV_ENV"
        if os.environ.get(gov_env_var_name) != "True":
            gov_cloud_error_message = GOV_CLOUD_CONSENT_FORMULA + (
                "\n"
                "To confirm your consent, set the environment variable "
                f"`{gov_env_var_name}` to `“True”`."
            )
            raise GovCloudEnvironmentConsentError(gov_cloud_error_message)
        gov_cloud_warn_msg = (
            f"\nYou set the environment variable `{gov_env_var_name}` to `'True'`"
            " which means that you consent to the disclosure of information below."
            "\n"
        ) + GOV_CLOUD_CONSENT_FORMULA

        logger.warning(gov_cloud_warn_msg)


class AsyncFileReader(httpx.AsyncByteStream):
    """File reader for async httpx requests."""

    def __init__(self, file_path: str | Path, chunk_size: int = 8192) -> None:
        self.file_path = file_path
        self.chunk_size = chunk_size

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        with open(self.file_path, "rb") as file:
            async_file = AsyncFile(file)
            while chunk := await async_file.read(self.chunk_size):
                yield chunk

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AsyncFileReader):
            return False

        return self.file_path == other.file_path and self.chunk_size == other.chunk_size

    def __repr__(self):
        return (
            f"AsyncFileReader(file_path={self.file_path}, chunk_size={self.chunk_size})"
        )


def get_from_json(json_object: Any, key_chain: list[Any], default: Any = None) -> Any:
    try:
        for key in key_chain:
            json_object = json_object[key]
    except (LookupError, TypeError):
        return default

    return json_object
