#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import os
import re

TRAINING_MODEL_HREF_PATTERN = "{}/v4/trainings/{}"
TRAINING_MODELS_HREF_PATTERN = "{}/v4/trainings"
REPO_MODELS_FRAMEWORKS_HREF_PATTERN = "{}/v3/models/frameworks"

INSTANCE_ENDPOINT_HREF_PATTERN = "{}/v3/wml_instance"
INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN = "{}/v3/wml_instances/{}"
TOKEN_ENDPOINT_HREF_PATTERN = "{}/v3/identity/token"
CPD_TOKEN_ENDPOINT_HREF_PATTERN = "{}/icp4d-api/v1/authorize"
CPD_BEDROCK_TOKEN_ENDPOINT_HREF_PATTERN = "{}/idprovider/v1/auth/identitytoken"
CPD_VALIDATION_TOKEN_ENDPOINT_HREF_PATTERN = "{}/v1/preauth/validateAuth"
CPD_PUBLIC_KEYS_ENDPOINT_HREF_PATTERN = "{}/auth/jwks"
EXPERIMENTS_HREF_PATTERN = "{}/v4/experiments"
EXPERIMENT_HREF_PATTERN = "{}/v4/experiments/{}"
EXPERIMENT_RUNS_HREF_PATTERN = "{}/v3/experiments/{}/runs"
EXPERIMENT_RUN_HREF_PATTERN = "{}/v3/experiments/{}/runs/{}"

PUBLISHED_MODEL_HREF_PATTERN = "{}/v4/models/{}"
PUBLISHED_MODEL_CONTENT_HREF_PATTERN = "{}/v4/models/{}/content"
PUBLISHED_MODELS_HREF_PATTERN = "{}/v4/models"
LEARNING_CONFIGURATION_HREF_PATTERN = (
    "{}/v3/wml_instances/{}/published_models/{}/learning_configuration"
)
LEARNING_ITERATION_HREF_PATTERN = (
    "{}/v3/wml_instances/{}/published_models/{}/learning_iterations/{}"
)
LEARNING_ITERATIONS_HREF_PATTERN = (
    "{}/v3/wml_instances/{}/published_models/{}/learning_iterations"
)
EVALUATION_METRICS_HREF_PATTERN = (
    "{}/v3/wml_instances/{}/published_models/{}/evaluation_metrics"
)
FEEDBACK_HREF_PATTERN = "{}/v3/wml_instances/{}/published_models/{}/feedback"

DEPLOYMENTS_HREF_PATTERN = "{}/v4/deployments"
DEPLOYMENT_HREF_PATTERN = "{}/v4/deployments/{}"
DEPLOYMENT_PREDICTIONS_HREF_PATTERN = "{}/v4/deployments/{}/predictions"
DEPLOYMENT_AI_SERVICE_HREF_PATTERN = "{}/v4/deployments/{}/ai_service"
DEPLOYMENT_AI_SERVICE_STREAM_HREF_PATTERN = "{}/v4/deployments/{}/ai_service_stream"
DEPLOYMENT_JOB_HREF_PATTERN = "{}/v4/deployment_jobs"
DEPLOYMENT_JOBS_HREF_PATTERN = "{}/v4/deployment_jobs/{}"
DEPLOYMENT_ENVS_HREF_PATTERN = "{}/v4/deployments/environments"
DEPLOYMENT_ENV_HREF_PATTERN = "{}/v4/deployments/environments/{}"

MODEL_LAST_VERSION_HREF_PATTERN = "{}/v4/models/{}"
MODEL_DOWNLOAD_HREF_PATTERN = "{}/v4/models/{}/download"
DEFINITION_HREF_PATTERN = "{}/v3/ml_assets/training_definitions/{}"
DEFINITIONS_HREF_PATTERN = "{}/v3/ml_assets/training_definitions"

FUNCTION_HREF_PATTERN = "{}/v4/functions/{}"
FUNCTION_CODE_HREF_PATTERN = "{}/v4/functions/{}/code"
FUNCTION_REVISIONS_HREF_PATTERN = "{}/v4/functions/{}/revisions"
FUNCTION_LATEST_CONTENT_HREF_PATTERN = "{}/v4/functions/{}/content"
FUNCTIONS_HREF_PATTERN = "{}/v4/functions"

AI_SERVICE_HREF_PATTERN = "{}/v4/ai_services/{}"
AI_SERVICE_CODE_HREF_PATTERN = "{}/v4/ai_services/{}/code"
AI_SERVICE_REVISIONS_HREF_PATTERN = "{}/v4/ai_services/{}/revisions"
AI_SERVICES_LATEST_CONTENT_HREF_PATTERN = "{}/v4/ai_services/{}/content"
AI_SERVICES_HREF_PATTERN = "{}/v4/ai_services"

IAM_TOKEN_API = "{}&grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey"
IAM_TOKEN_URL = "{}/identity/token"
AWS_TOKEN_URL = "{}/api/2.0/apikeys/token"
IAM_PUBLIC_KEYS_URL = "{}/identity/keys"
AWS_PUBLIC_KEYS_URL = "{}/api/2.0/jwks"
PROD_SVT_URL = [
    "https://ca-tor.ml.cloud.ibm.com",
    "https://private.ca-tor.ml.cloud.ibm.com",
    "https://wxai-qa.ml.cloud.ibm.com",
    "https://private.wxai-qa.ml.cloud.ibm.com",
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-gb.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com",
    "https://ap-south-1.aws.wxai.ibm.com",
    "https://wxai.ibmforusgov.com",
    "https://wxai.prep.ibmforusgov.com",
    "https://ibm-watson-ml.mybluemix.net",
    "https://ibm-watson-ml.eu-gb.bluemix.net",
    "https://private.us-south.ml.cloud.ibm.com",
    "https://private.eu-gb.ml.cloud.ibm.com",
    "https://private.eu-de.ml.cloud.ibm.com",
    "https://private.jp-tok.ml.cloud.ibm.com",
    "https://private.au-syd.ml.cloud.ibm.com",
    "https://private.ap-south-1.aws.wxai.ibm.com",
    "https://private.wxai.ibmforusgov.com",
    "https://private.wxai.prep.ibmforusgov.com",
    "https://yp-qa.ml.cloud.ibm.com",
    "https://private.yp-qa.ml.cloud.ibm.com",
    "https://yp-cr.ml.cloud.ibm.com",
    "https://private.yp-cr.ml.cloud.ibm.com",
]

PIPELINES_HREF_PATTERN = "{}/v4/pipelines"
PIPELINE_HREF_PATTERN = "{}/v4/pipelines/{}"


SPACES_HREF_PATTERN = "{}/v4/spaces"
SPACE_HREF_PATTERN = "{}/v4/spaces/{}"
MEMBER_HREF_PATTERN = "{}/v4/spaces/{}/members/{}"
MEMBERS_HREF_PATTERN = "{}/v4/spaces/{}/members"

SPACES_PLATFORM_HREF_PATTERN = "{}/v2/spaces"
SPACE_PLATFORM_HREF_PATTERN = "{}/v2/spaces/{}"
SPACES_MEMBERS_HREF_PATTERN = "{}/v2/spaces/{}/members"
SPACES_MEMBER_HREF_PATTERN = "{}/v2/spaces/{}/members/{}"

PROJECT = "{}/v2/projects/{}"
PROJECTS = "{}/v2/projects"
TRANSACTIONAL_PROJECT = "{}/transactional/v2/projects/{}"
TRANSACTIONAL_PROJECTS = "{}/transactional/v2/projects"
PROJECTS_MEMBERS_HREF_PATTERN = "{}/v2/projects/{}/members"
PROJECTS_MEMBER_HREF_PATTERN = "{}/v2/projects/{}/members/{}"

V4_INSTANCE_ID_HREF_PATTERN = "{}/ml/v4/instances/{}"

API_VERSION = "/v4"
SPACES = "/spaces"
PIPELINES = "/pipelines"
EXPERIMENTS = "/experiments"
LIBRARIES = "/libraries"
RUNTIMES = "/runtimes"
SOFTWARE_SPEC = "/software_specifications"
DEPLOYMENTS = "/deployments"
ASSET = "{}/v2/assets/{}"
ASSETS = "{}/v2/assets"
ASSET_TYPE = "{}/v2/asset_types"
ASSET_FILES = "{}/v2/asset_files/"
FOLDER_ASSET = "{}/v2/folder_assets/{}"
FOLDER_ASSETS = "{}/v2/folder_assets"
TRASHED_ASSETS = "{}/v2/trashed_assets"
TRASHED_ASSETS_PURGE_ALL = "{}/v2/trashed_assets/purge_all"
TRASHED_ASSET = "{}/v2/trashed_assets/{}"
TRASHED_ASSET_RESTORE = "{}/v2/trashed_assets/{}/restore"
ATTACHMENT = "{}/v2/assets/{}/attachments/{}"
ATTACHMENT_COMPLETE = "{}/v2/assets/{}/attachments/{}/complete"
ATTACHMENTS = "{}/v2/assets/{}/attachments"
SEARCH_ASSETS = "{}/v2/asset_types/{}/search"
SEARCH_MODEL_DEFINITIONS = "{}/v2/asset_types/wml_model_definition/search"
SEARCH_DATA_ASSETS = "{}/v2/asset_types/data_asset/search"
SEARCH_FOLDER_ASSETS = "{}/v2/asset_types/folder_asset/search"
SEARCH_SHINY = "{}/v2/asset_types/shiny_asset/search"
SEARCH_SCRIPT = "{}/v2/asset_types/script/search"
GIT_BASED_PROJECT_ASSET = "{}/userfs/v2/assets/{}"
GIT_BASED_PROJECT_ASSETS = "{}/userfs/v2/assets"
GIT_BASED_PROJECT_ASSET_TYPE = "{}/userfs/v2/asset_types"
GIT_BASED_PROJECT_ASSET_FILES = "{}/v2/asset_files/"
GIT_BASED_PROJECT_FOLDER_ASSET = "{}/userfs/v2/folder_assets/{}"
GIT_BASED_PROJECT_FOLDER_ASSETS = "{}/userfs/v2/folder_assets"
GIT_BASED_PROJECT_ATTACHMENT = "{}/userfs/v2/assets/{}/attachments/{}"
GIT_BASED_PROJECT_ATTACHMENT_COMPLETE = "{}/userfs/v2/assets/{}/attachments/{}/complete"
GIT_BASED_PROJECT_ATTACHMENTS = "{}/userfs/v2/assets/{}/attachments"
GIT_BASED_PROJECT_SEARCH_ASSETS = "{}/userfs/v2/asset_types/{}/search"
GIT_BASED_PROJECT_SEARCH_MODEL_DEFINITIONS = (
    "{}/userfs/v2/asset_types/wml_model_definition/search"
)
GIT_BASED_PROJECT_SEARCH_DATA_ASSETS = "{}/userfs/v2/asset_types/data_asset/search"
GIT_BASED_PROJECT_SEARCH_FOLDER_ASSETS = "{}/userfs/v2/asset_types/folder_asset/search"
GIT_BASED_PROJECT_SEARCH_SHINY = "{}/userfs/v2/asset_types/shiny_asset/search"
GIT_BASED_PROJECT_SEARCH_SCRIPT = "{}/userfs/v2/asset_types/script/search"
DATA_SOURCE_TYPES = "{}/v2/datasource_types"
DATA_SOURCE_TYPE = "{}/v2/datasource_types/{}"
CONNECTION_ASSET = "{}/v2/connections"
CONNECTION_ASSET_SEARCH = "{}/v2/connections"
CONNECTION_BY_ID = "{}/v2/connections/{}"
CONNECTIONS_FILES = "{}/v2/connections/files"
CONNECTIONS_FILE = "{}/v2/connections/files/{}"
SOFTWARE_SPECIFICATION = "{}/v2/software_specifications/{}"
SOFTWARE_SPECIFICATIONS = "{}/v2/software_specifications"
SOFTWARE_SPECIFICATION_PACKAGE_EXTENSION = (
    "{}/v2/software_specifications/{}/package_extensions/{}"
)
HARDWARE_SPECIFICATION = "{}/v2/hardware_specifications/{}"
HARDWARE_SPECIFICATIONS = "{}/v2/hardware_specifications"
PACKAGE_EXTENSION = "{}/v2/package_extensions/{}"
PACKAGE_EXTENSION_UPLOAD_COMPLETE = "{}/v2/package_extensions/{}/upload_complete"
PACKAGE_EXTENSIONS = "{}/v2/package_extensions"
PARAMETER_SET = "{}/v2/parameter_sets/{}"
PARAMETER_SETS = "{}/v2/parameter_sets"
RUNTIME_DEFINITION = "{}/v2/runtime_definitions/{}"
RUNTIME_DEFINITIONS = "{}/v2/runtime_definitions"
JOBS_RUNS = "{}/v2/jobs/{}/runs/{}"

V4GA_CLOUD_MIGRATION = "{}/ml/v4/repository"
V4GA_CLOUD_MIGRATION_ID = "{}/ml/v4/repository/{}"

REMOTE_TRAINING_SYSTEM = "{}/v4/remote_training_systems"
REMOTE_TRAINING_SYSTEM_ID = "{}/v4/remote_training_systems/{}"

FM_CHAT = "{}/ml/v1/text/{}"
FM_GENERATION = "{}/ml/v1/text/generation"
FM_GENERATION_BETA = "{}/ml/v1-beta/generation/{}"  # Remove on CPD 5.0 release
FM_GENERATION_STREAM = "{}/ml/v1/text/generation_stream"
FM_GET_SPECS = "{}/ml/v1/foundation_model_specs"
FM_GET_SPECS_BETA = "{}/ml/v1-beta/foundation_model_specs?version=2022-08-01"  # Remove on CPD 5.0 release
FM_GET_CUSTOM_FOUNDATION_MODELS = "{}/ml/v4/custom_foundation_models"
FM_GET_TASKS = "{}/ml/v1/foundation_model_tasks?limit={}"
FM_TOKENIZE = "{}/ml/v1/text/tokenization"
FM_TOKENIZE_BETA = "{}/ml/v1-beta/text/tokenization"  # Remove on CPD 5.0 release
FM_EMBEDDINGS = "{}/ml/v1/text/embeddings"
FM_TIME_SERIES = "{}/ml/v1/time_series/forecast"
FM_AUDIO_TRANSCRIPTIONS = "{}/ml/v1/audio/transcriptions"

AUTOAI_RAG = "{}/ml/v1/autoai/rags"
AUTOAI_RAG_ID = "{}/ml/v1/autoai/rags/{}"

FM_DEPLOYMENT_GENERATION = "{}/ml/v1/deployments/{}/text/generation"
FM_DEPLOYMENT_GENERATION_STREAM = "{}/ml/v1/deployments/{}/text/generation_stream"
FM_DEPLOYMENT_CHAT = "{}/ml/v1/deployments/{}/text/chat"
FM_DEPLOYMENT_CHAT_STREAM = "{}/ml/v1/deployments/{}/text/chat_stream"
FM_DEPLOYMENT_GENERATION_BETA = (
    "{}/ml/v1-beta/deployments/{}/generation/{}"  # Remove on CPD 5.0 release
)
FM_DEPLOYMENT_TIME_SERIES = "{}/ml/v1/deployments/{}/time_series/forecast"

AI_SERVICES_DEPLOYMENT_GENERATION = "{}/ml/v4/deployments/{}/ai_service"
AI_SERVICES_DEPLOYMENT_GENERATION_STREAM = "{}/ml/v4/deployments/{}/ai_service_stream"
FM_FINE_TUNING = "{}/ml/v1/fine_tunings/{}"
FM_FINE_TUNINGS = "{}/ml/v1/fine_tunings"

PROMPTS = "{}/wx/v1-beta/prompts"
GA_PROMPTS = "{}/wx/v1/prompts"
PROMPTS_GET_ALL = "{}/v2/asset_types/wx_prompt/search"

TEXT_DETECTION = "{}/ml/v1/text/detection"

TEXT_EXTRACTIONS = "{}/ml/v1/text/extractions"
TEXT_EXTRACTION = "{}/ml/v1/text/extractions/{}"

RERANK = "{}/ml/v1/text/rerank"

EXPORTS = "{}/v2/asset_exports"
EXPORT_ID = "{}/v2/asset_exports/{}"
EXPORT_ID_CONTENT = "{}/v2/asset_exports/{}/content"

IMPORTS = "{}/v2/asset_imports"
IMPORT_ID = "{}/v2/asset_imports/{}"

VOLUMES = "{}/zen-data/v3/service_instances"
VOLUME_ID = "{}/zen-data/v3/service_instances/{}"
VOLUME_SERVICE = "{}/zen-data/v1/volumes/volume_services/{}"
VOLUME_SERVICE_FILE_UPLOAD = "{}/zen-volumes/{}/v1/volumes/files/"
VOLUME_MONITOR = "{}/zen-volumes/{}/v1/monitor"

PROMOTE_ASSET = "{}/projects/api/rest/catalogs/assets/{}/promote"

WKC_MODEL_REGISTER = "{}/v1/aigov/model_inventory/models/{}/model_entry"
WKC_MODEL_LIST_FROM_CATALOG = "{}/v1/aigov/model_inventory/{}/model_entries"
WKC_MODEL_LIST_ALL = "{}/v1/aigov/model_inventory/model_entries"
TASK_CREDENTIALS = "{}/v1/task_credentials/{}"
TASK_CREDENTIALS_ALL = "{}/v1/task_credentials"

TAXONOMY = "{}/ml/v4/taxonomies/{}"
TAXONOMIES_IMPORTS = "{}/ml/v1/tuning/taxonomies_imports"
TAXONOMIES_IMPORT = "{}/ml/v1/tuning/taxonomies_imports/{}"
DOCUMENT_EXTRACTIONS = "{}/ml/v1/tuning/documents"
DOCUMENT_EXTRACTION = "{}/ml/v1/tuning/documents/{}"
SYNTHETIC_DATA_GENERATIONS = "{}/ml/v1/tuning/synthetic_data"
SYNTHETIC_DATA_GENERATION = "{}/ml/v1/tuning/synthetic_data/{}"

UTILITY_AGENT_TOOLS_BETA = "{}/wx/v1-beta/utility_agent_tools"
UTILITY_AGENT_TOOLS_RUN_BETA = "{}/wx/v1-beta/utility_agent_tools/run"

VECTOR_INDEXES = "{}/wx/v1/vector_indexes"
VECTOR_INDEX = "{}/wx/v1/vector_indexes/{}"

VECTOR_INDEXES_GET_ALL = "{}/v2/asset_types/vector_index/search"
# AI GATEWAY
GATEWAY_TENANT = "{}/ml/gateway/v1/tenant"
GATEWAY_PROVIDERS = "{}/ml/gateway/v1/providers"
GATEWAY_PROVIDER = "{}/ml/gateway/v1/providers/{}"
GATEWAY_PROVIDER_AVAILABLE_MODELS = "{}/ml/gateway/v1/providers/{}/models/available"
GATEWAY_UPDATE_PROVIDER = "{}/ml/gateway/v1/providers/{}/{}"
GATEWAY_MODELS = "{}/ml/gateway/v1/providers/{}/models"
GATEWAY_ALL_TENANT_MODELS = "{}/ml/gateway/v1/models"
GATEWAY_MODEL = "{}/ml/gateway/v1/models/{}"
GATEWAY_POLICIES = "{}/ml/gateway/v1/policies"
GATEWAY_EMBEDDINGS = "{}/ml/gateway/v1/embeddings"
GATEWAY_TEXT_COMPLETIONS = "{}/ml/gateway/v1/completions"
GATEWAY_CHAT_COMPLETIONS = "{}/ml/gateway/v1/chat/completions"


def is_url(s: str) -> bool:
    res = re.match(r"https?:\/\/.+", s)
    return res is not None


def is_id(s: str) -> bool:
    res = re.match(r"[a-z0-9\-]{36}", s)
    return res is not None


class HrefDefinitions:
    def __init__(
        self,
        url: str,
        instance_id: str,
        version: float | None,
        bedrock_url: str | None,
        cloud_platform_spaces: bool,
        cp4d_platform_spaces: bool,
        platform_url: str,
        project_type: str,
        _use_fm_ga_api: bool,
    ):
        self.url = url
        self.instance_id = instance_id
        self.version = version
        self.bedrock_url = bedrock_url
        self.cloud_platform_spaces = cloud_platform_spaces
        self.cp4d_platform_spaces = cp4d_platform_spaces
        self.platform_url = platform_url
        self.project_type = project_type
        self._use_fm_ga_api = _use_fm_ga_api
        self.prepend = "/ml"

    def _is_git_based_project(self) -> bool:
        return self.project_type == "local_git_storage"

    def _get_platform_url_if_exists(self) -> str:
        return self.platform_url if self.platform_url else self.url

    def get_training_href(self, model_id: str) -> str:
        return TRAINING_MODEL_HREF_PATTERN.format(self.url + self.prepend, model_id)

    def get_trainings_href(self) -> str:
        return TRAINING_MODELS_HREF_PATTERN.format(self.url + self.prepend)

    def get_repo_models_frameworks_href(self) -> str:
        return REPO_MODELS_FRAMEWORKS_HREF_PATTERN.format(self.url + self.prepend)

    def get_instance_endpoint_href(self) -> str:
        return INSTANCE_ENDPOINT_HREF_PATTERN.format(self.url)

    def get_instance_by_id_endpoint_href(self) -> str:
        return INSTANCE_BY_ID_ENDPOINT_HREF_PATTERN.format(self.url, self.instance_id)

    def get_token_endpoint_href(self) -> str:
        return TOKEN_ENDPOINT_HREF_PATTERN.format(self.url)

    def get_cpd_token_endpoint_href(self) -> str:
        return CPD_TOKEN_ENDPOINT_HREF_PATTERN.format(
            self.url.replace(":31002", ":31843")
        )

    def get_cpd_bedrock_token_endpoint_href(self) -> str:
        return CPD_BEDROCK_TOKEN_ENDPOINT_HREF_PATTERN.format(self.bedrock_url)

    def get_cpd_validation_token_endpoint_href(self) -> str:
        return CPD_VALIDATION_TOKEN_ENDPOINT_HREF_PATTERN.format(self.url)

    def get_cpd_public_keys_endpoint_href(self) -> str:
        return CPD_PUBLIC_KEYS_ENDPOINT_HREF_PATTERN.format(self.url)

    def get_published_model_href(self, model_id: str) -> str:
        return PUBLISHED_MODEL_HREF_PATTERN.format(self.url + self.prepend, model_id)

    def get_published_model_content_href(self, model_id: str) -> str:
        return PUBLISHED_MODEL_CONTENT_HREF_PATTERN.format(
            self.url + self.prepend, model_id
        )

    def get_published_models_href(self) -> str:
        return PUBLISHED_MODELS_HREF_PATTERN.format(self.url + self.prepend)

    def get_learning_configuration_href(self, model_id: str) -> str:
        return LEARNING_CONFIGURATION_HREF_PATTERN.format(
            self.url, self.instance_id, model_id
        )

    def get_learning_iterations_href(self, model_id: str) -> str:
        return LEARNING_ITERATIONS_HREF_PATTERN.format(
            self.url, self.instance_id, model_id
        )

    def get_learning_iteration_href(self, model_id: str, iteration_id: str) -> str:
        return LEARNING_ITERATION_HREF_PATTERN.format(
            self.url,
            self.instance_id,
            model_id,
            iteration_id,
        )

    def get_evaluation_metrics_href(self, model_id: str) -> str:
        return EVALUATION_METRICS_HREF_PATTERN.format(
            self.url, self.instance_id, model_id
        )

    def get_feedback_href(self, model_id: str) -> str:
        return FEEDBACK_HREF_PATTERN.format(self.url, self.instance_id, model_id)

    def get_model_last_version_href(self, artifact_id: str) -> str:
        return MODEL_LAST_VERSION_HREF_PATTERN.format(
            self.url + self.prepend, artifact_id
        )

    def get_model_download_href(self, artifact_id: str) -> str:
        return MODEL_DOWNLOAD_HREF_PATTERN.format(self.url + self.prepend, artifact_id)

    def get_deployments_href(self) -> str:
        return DEPLOYMENTS_HREF_PATTERN.format(self.url + self.prepend)

    def get_experiments_href(self) -> str:
        return EXPERIMENTS_HREF_PATTERN.format(self.url + self.prepend)

    def get_experiment_href(self, experiment_id: str) -> str:
        return EXPERIMENT_HREF_PATTERN.format(self.url + self.prepend, experiment_id)

    def get_experiment_runs_href(self, experiment_id: str) -> str:
        return EXPERIMENT_RUNS_HREF_PATTERN.format(self.url, experiment_id)

    def get_experiment_run_href(
        self, experiment_id: str, experiment_run_id: str
    ) -> str:
        return EXPERIMENT_RUN_HREF_PATTERN.format(
            self.url, experiment_id, experiment_run_id
        )

    def get_deployment_href(self, deployment_id: str) -> str:
        return DEPLOYMENT_HREF_PATTERN.format(self.url + self.prepend, deployment_id)

    def get_deployment_predictions_href(self, deployment_id: str) -> str:
        return DEPLOYMENT_PREDICTIONS_HREF_PATTERN.format(
            self.url + self.prepend, deployment_id
        )

    def get_deployment_ai_service_href(self, deployment_id: str) -> str:
        return DEPLOYMENT_AI_SERVICE_HREF_PATTERN.format(
            self.url + self.prepend, deployment_id
        )

    def get_deployment_ai_service_stream_href(self, deployment_id: str) -> str:
        return DEPLOYMENT_AI_SERVICE_STREAM_HREF_PATTERN.format(
            self.url + self.prepend, deployment_id
        )

    def get_definition_href(self, definition_id: str) -> str:
        return DEFINITION_HREF_PATTERN.format(self.url, definition_id)

    def get_definitions_href(self) -> str:
        return DEFINITIONS_HREF_PATTERN.format(self.url)

    def get_function_href(self, ai_function_id: str) -> str:
        return FUNCTION_HREF_PATTERN.format(self.url + self.prepend, ai_function_id)

    def get_function_code_href(self, ai_function_id: str) -> str:
        return FUNCTION_CODE_HREF_PATTERN.format(
            self.url + self.prepend, ai_function_id
        )

    def get_function_revisions_href(self, ai_function_id: str) -> str:
        return FUNCTION_REVISIONS_HREF_PATTERN.format(
            self.url + self.prepend, ai_function_id
        )

    def get_function_latest_revision_content_href(self, ai_function_id: str) -> str:
        return FUNCTION_LATEST_CONTENT_HREF_PATTERN.format(self.url, ai_function_id)

    def get_functions_href(self) -> str:
        return FUNCTIONS_HREF_PATTERN.format(self.url + self.prepend)

    def get_ai_service_href(self, ai_service_id: str) -> str:
        return AI_SERVICE_HREF_PATTERN.format(self.url + self.prepend, ai_service_id)

    def get_ai_service_code_href(self, ai_service_id: str) -> str:
        return AI_SERVICE_CODE_HREF_PATTERN.format(
            self.url + self.prepend, ai_service_id
        )

    def get_ai_service_revisions_href(self, ai_service_id: str) -> str:
        return AI_SERVICE_REVISIONS_HREF_PATTERN.format(
            self.url + self.prepend, ai_service_id
        )

    def get_ai_services_latest_revision_content_href(self, ai_service_id: str) -> str:
        return AI_SERVICES_LATEST_CONTENT_HREF_PATTERN.format(self.url, ai_service_id)

    def get_ai_services_href(self) -> str:
        return AI_SERVICES_HREF_PATTERN.format(self.url + self.prepend)

    def get_pipeline_href(self, pipeline_id: str) -> str:
        return PIPELINE_HREF_PATTERN.format(self.url + self.prepend, pipeline_id)

    def get_pipelines_href(self) -> str:
        return PIPELINES_HREF_PATTERN.format(self.url + self.prepend)

    def get_space_href(self, spaces_id: str) -> str:
        return SPACE_HREF_PATTERN.format(self.url, spaces_id)

    def get_spaces_href(self) -> str:
        return SPACES_HREF_PATTERN.format(self.url)

    def get_platform_space_href(self, spaces_id: str) -> str:
        return SPACE_PLATFORM_HREF_PATTERN.format(
            self._get_platform_url_if_exists(), spaces_id
        )

    def get_platform_spaces_href(self) -> str:
        return SPACES_PLATFORM_HREF_PATTERN.format(self._get_platform_url_if_exists())

    def get_platform_spaces_member_href(self, spaces_id: str, member_id: str) -> str:
        return SPACES_MEMBER_HREF_PATTERN.format(
            self._get_platform_url_if_exists(), spaces_id, member_id
        )

    def get_platform_spaces_members_href(self, spaces_id: str) -> str:
        return SPACES_MEMBERS_HREF_PATTERN.format(
            self._get_platform_url_if_exists(), spaces_id
        )

    def get_projects_member_href(self, project_id: str, member_id: str) -> str:
        return PROJECTS_MEMBER_HREF_PATTERN.format(
            self._get_platform_url_if_exists(), project_id, member_id
        )

    def get_projects_members_href(self, project_id: str) -> str:
        return PROJECTS_MEMBERS_HREF_PATTERN.format(
            self._get_platform_url_if_exists(), project_id
        )

    def get_v4_instance_id_href(self, instance_id: str) -> str:
        return V4_INSTANCE_ID_HREF_PATTERN.format(self.url, instance_id)

    def get_async_deployment_job_href(self) -> str:
        return DEPLOYMENT_JOB_HREF_PATTERN.format(self.url + self.prepend)

    def get_async_deployment_jobs_href(self, job_id: str) -> str:
        return DEPLOYMENT_JOBS_HREF_PATTERN.format(self.url + self.prepend, job_id)

    def get_iam_token_api(self, apikey) -> str:
        return IAM_TOKEN_API.format(apikey)

    def get_aws_token_url(self) -> str:
        # On AWS GovCloud PreProd & Prod IAM endpoints are not available from outside, because of this,
        # normal path is available internally for services  by setting WATSONX_USE_PRIVATE_TOKEN_URL=true,
        # while the users received new endpoints `/api/rest/mcsp/apikeys/token`, with the same usage as original one.

        match self.url:
            case (
                "https://ap-south-1.aws.wxai.ibm.com"
                | "https://private.ap-south-1.aws.wxai.ibm.com"
            ):
                # Mumbai (AWS)
                base_auth_url = "https://account-iam.platform.saas.ibm.com"
            case (
                "https://wxai.prep.ibmforusgov.com"
                | "https://private.wxai.prep.ibmforusgov.com"
            ):
                # PreProd AWS GovCloud
                if (
                    os.getenv("WATSONX_USE_PRIVATE_TOKEN_URL", "").lower().strip()
                    == "true"
                ):  # path for internal services
                    base_auth_url = "https://account-iam.awsg.usge1.private.platform.prep.ibmforusgov.com"
                else:  # path for users
                    return "{}/api/rest/mcsp/apikeys/token".format(
                        self.platform_url.replace("https://api.", "https://")
                    )
            case (
                "https://wxai.ibmforusgov.com" | "https://private.wxai.ibmforusgov.com"
            ):
                # Prod AWS GovCloud
                if (
                    os.getenv("WATSONX_USE_PRIVATE_TOKEN_URL", "").lower().strip()
                    == "true"
                ):  # path for internal services
                    base_auth_url = "https://account-iam.awsg.usge1.private.platform.ibmforusgov.com"
                else:  # path for users
                    return "{}/api/rest/mcsp/apikeys/token".format(
                        self.platform_url.replace("https://api.", "https://")
                    )
            case _:
                # AWS Dev
                base_auth_url = "https://account-iam.platform.test.saas.ibm.com"

        return AWS_TOKEN_URL.format(base_auth_url)

    def get_aws_public_keys_url(self) -> str:
        return AWS_PUBLIC_KEYS_URL.format(
            "https://account-iam.platform.saas.ibm.com"
            if self.url in PROD_SVT_URL
            else "https://account-iam.platform.test.saas.ibm.com"
        )

    def get_iam_token_url(self) -> str:
        if self.url in PROD_SVT_URL:
            return IAM_TOKEN_URL.format("https://iam.cloud.ibm.com")
        else:
            return IAM_TOKEN_URL.format("https://iam.test.cloud.ibm.com")

    def get_iam_public_keys_url(self) -> str:
        if self.url in PROD_SVT_URL:
            return IAM_PUBLIC_KEYS_URL.format("https://iam.cloud.ibm.com")
        else:
            return IAM_PUBLIC_KEYS_URL.format("https://iam.test.cloud.ibm.com")

    def get_member_href(self, spaces_id: str, member_id: str) -> str:
        return MEMBER_HREF_PATTERN.format(self.url, spaces_id, member_id)

    def get_members_href(self, spaces_id: str) -> str:
        return MEMBERS_HREF_PATTERN.format(self.url, spaces_id)

    def get_data_asset_href(self, asset_id: str) -> str:
        return (
            ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET
        ).format(self._get_platform_url_if_exists(), asset_id)

    def get_data_assets_href(self) -> str:
        return (
            ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_folder_asset_href(self, folder_asset_id: str) -> str:
        return (
            FOLDER_ASSET
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_FOLDER_ASSET
        ).format(self._get_platform_url_if_exists(), folder_asset_id)

    def get_folder_assets_href(self) -> str:
        return (
            FOLDER_ASSETS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_FOLDER_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_assets_href(self) -> str:
        return (
            ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_asset_href(self, asset_id: str) -> str:
        return (
            ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET
        ).format(self._get_platform_url_if_exists(), asset_id)

    def get_base_asset_href(self, asset_id: str) -> str:
        return (
            ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET
        ).format("", asset_id)

    def get_base_assets_href(self) -> str:
        return (
            ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS
        ).format("")

    def get_base_asset_with_type_href(self, asset_type: str, asset_id: str) -> str:
        return (
            (
                ASSET if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSET
            ).format("", asset_type)
            + "/"
            + asset_id
        )

    def get_attachment_href(self, asset_id: str, attachment_id: str) -> str:
        return (
            ATTACHMENT
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_ATTACHMENT
        ).format(self._get_platform_url_if_exists(), asset_id, attachment_id)

    def get_attachments_href(self, asset_id: str) -> str:
        return (
            ATTACHMENTS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_ATTACHMENTS
        ).format(self._get_platform_url_if_exists(), asset_id)

    def get_attachment_complete_href(self, asset_id: str, attachment_id: str) -> str:
        return (
            ATTACHMENT_COMPLETE
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_ATTACHMENT_COMPLETE
        ).format(self._get_platform_url_if_exists(), asset_id, attachment_id)

    def get_search_data_asset_href(self) -> str:
        return (
            SEARCH_DATA_ASSETS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_DATA_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_search_folder_asset_href(self) -> str:
        return (
            SEARCH_FOLDER_ASSETS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_FOLDER_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_search_shiny_href(self) -> str:
        return (
            SEARCH_SHINY
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_SHINY
        ).format(self._get_platform_url_if_exists())

    def get_search_script_href(self) -> str:
        return (
            SEARCH_SCRIPT
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_SCRIPT
        ).format(self._get_platform_url_if_exists())

    def get_model_definition_assets_href(self) -> str:
        return (
            ASSETS if not self._is_git_based_project() else GIT_BASED_PROJECT_ASSETS
        ).format(self._get_platform_url_if_exists())

    def get_model_definition_search_asset_href(self) -> str:
        return (
            SEARCH_MODEL_DEFINITIONS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_MODEL_DEFINITIONS
        ).format(self._get_platform_url_if_exists())

    # note: leave `wsd` in name since APIClient is still wrapped
    def get_wsd_model_attachment_href(self) -> str:
        return (
            ASSET_FILES
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_ASSET_FILES
        ).format(self.url)

    def get_asset_search_href(self, asset_type: str) -> str:
        return (
            SEARCH_ASSETS
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_SEARCH_ASSETS
        ).format(self._get_platform_url_if_exists(), asset_type)

    def get_wsd_asset_type_href(self) -> str:
        return (
            ASSET_TYPE
            if not self._is_git_based_project()
            else GIT_BASED_PROJECT_ASSET_TYPE
        ).format(self.url)

    def get_trashed_assets_href(self) -> str:
        return TRASHED_ASSETS.format(self.url)

    def get_trashed_assets_purge_all_href(self) -> str:
        return TRASHED_ASSETS_PURGE_ALL.format(self.url)

    def get_trashed_asset_href(self, asset_id: str) -> str:
        return TRASHED_ASSET.format(self.url, asset_id)

    def get_trashed_asset_restore_href(self, asset_id: str) -> str:
        return TRASHED_ASSET_RESTORE.format(self.url, asset_id)

    def get_connections_href(self) -> str:
        return CONNECTION_ASSET.format(self._get_platform_url_if_exists())

    def get_connection_by_id_href(self, connection_id: str) -> str:
        return CONNECTION_BY_ID.format(
            self._get_platform_url_if_exists(), connection_id
        )

    def get_connections_files_href(self) -> str:
        return CONNECTIONS_FILES.format(self._get_platform_url_if_exists())

    def get_connections_file_href(self, file_name: str) -> str:
        return CONNECTIONS_FILE.format(self._get_platform_url_if_exists(), file_name)

    def get_connection_data_types_href(self) -> str:
        return DATA_SOURCE_TYPES.format(self._get_platform_url_if_exists())

    def get_connection_data_type_href(self, datasource_type: str) -> str:
        return DATA_SOURCE_TYPE.format(
            self._get_platform_url_if_exists(), datasource_type
        )

    def get_sw_spec_href(self, sw_spec_id: str) -> str:
        return SOFTWARE_SPECIFICATION.format(
            self._get_platform_url_if_exists(), sw_spec_id
        )

    def get_sw_specs_href(self) -> str:
        return SOFTWARE_SPECIFICATIONS.format(self._get_platform_url_if_exists())

    def get_sw_spec_pkg_extn_href(self, sw_spec_id: str, pkg_extn_id: str):
        return SOFTWARE_SPECIFICATION_PACKAGE_EXTENSION.format(
            self._get_platform_url_if_exists(), sw_spec_id, pkg_extn_id
        )

    def get_hw_spec_href(self, hw_spec_id: str) -> str:
        return HARDWARE_SPECIFICATION.format(
            self._get_platform_url_if_exists(), hw_spec_id
        )

    def get_hw_specs_href(self) -> str:
        return HARDWARE_SPECIFICATIONS.format(self._get_platform_url_if_exists())

    def get_pkg_extn_href(self, pkg_extn_id: str) -> str:
        return PACKAGE_EXTENSION.format(self._get_platform_url_if_exists(), pkg_extn_id)

    def get_pkg_extn_upload_complete_href(self, pkg_extn_id: str) -> str:
        return PACKAGE_EXTENSION_UPLOAD_COMPLETE.format(
            self._get_platform_url_if_exists(), pkg_extn_id
        )

    def get_pkg_extns_href(self) -> str:
        return PACKAGE_EXTENSIONS.format(self._get_platform_url_if_exists())

    def get_project_href(self, project_id: str) -> str:
        return PROJECT.format(self._get_platform_url_if_exists(), project_id)

    def get_projects_href(self) -> str:
        return PROJECTS.format(self._get_platform_url_if_exists())

    def get_transactional_project_href(self, project_id: str) -> str:
        return TRANSACTIONAL_PROJECT.format(
            self._get_platform_url_if_exists(), project_id
        )

    def get_transactional_projects_href(self) -> str:
        return TRANSACTIONAL_PROJECTS.format(self._get_platform_url_if_exists())

    def v4ga_cloud_migration_href(self) -> str:
        return V4GA_CLOUD_MIGRATION.format(self.url)

    def v4ga_cloud_migration_id_href(self, migration_id: str) -> str:
        return V4GA_CLOUD_MIGRATION_ID.format(self.url, migration_id)

    def exports_href(self) -> str:
        return EXPORTS.format(self._get_platform_url_if_exists())

    def export_href(self, export_id: str) -> str:
        return EXPORT_ID.format(self._get_platform_url_if_exists(), export_id)

    def export_content_href(self, export_id: str) -> str:
        return EXPORT_ID_CONTENT.format(self._get_platform_url_if_exists(), export_id)

    def imports_href(self) -> str:
        return IMPORTS.format(self._get_platform_url_if_exists())

    def import_href(self, export_id: str) -> str:
        return IMPORT_ID.format(self._get_platform_url_if_exists(), export_id)

    def remote_training_systems_href(self) -> str:
        return REMOTE_TRAINING_SYSTEM.format(self.url + self.prepend)

    def remote_training_system_href(self, remote_training_systems_id: str) -> str:
        return REMOTE_TRAINING_SYSTEM_ID.format(
            self.url + self.prepend, remote_training_systems_id
        )

    def volumes_href(self) -> str:
        return VOLUMES.format(self.url)

    def volume_href(self, volume_id: str) -> str:
        return VOLUME_ID.format(self.url, volume_id)

    def volume_service_href(self, volume_name: str) -> str:
        return VOLUME_SERVICE.format(self.url, volume_name)

    def volume_upload_href(self, volume_name: str) -> str:
        return VOLUME_SERVICE_FILE_UPLOAD.format(self.url, volume_name)

    def volume_monitor_href(self, volume_name: str) -> str:
        return VOLUME_MONITOR.format(self.url, volume_name)

    def promote_asset_href(self, asset_id: str) -> str:
        if self.cloud_platform_spaces:
            data_platform_url = self.platform_url.replace("api.", "")
            return PROMOTE_ASSET.format(data_platform_url, asset_id)
        else:
            promote_href = PROMOTE_ASSET.format(self.url, asset_id)
            try:
                # note: For CPD older than 4.0 we need to roll back to older endpoint.
                if float(self.version) < 4.0:
                    promote_href = promote_href.replace("/projects", "")
                # --- end note
            finally:
                return promote_href

    def get_wkc_model_register_href(self, model_id: str) -> str:
        return WKC_MODEL_REGISTER.format(self._get_platform_url_if_exists(), model_id)

    def get_wkc_model_list_from_catalog_href(self, catalog_id: str) -> str:
        return WKC_MODEL_LIST_FROM_CATALOG.format(
            self._get_platform_url_if_exists(), catalog_id
        )

    def get_wkc_model_list_all_href(self) -> str:
        return WKC_MODEL_LIST_ALL.format(self._get_platform_url_if_exists())

    def get_wkc_model_delete_href(self, asset_id: str) -> str:
        return WKC_MODEL_REGISTER.format(self._get_platform_url_if_exists(), asset_id)

    def get_task_credentials_href(self, task_credentials_id: str) -> str:
        return TASK_CREDENTIALS.format(
            self._get_platform_url_if_exists(), task_credentials_id
        )

    def get_task_credentials_all_href(self) -> str:
        return TASK_CREDENTIALS_ALL.format(self._get_platform_url_if_exists())

    def get_fm_specifications_href(self) -> str:
        if self._use_fm_ga_api:
            return FM_GET_SPECS.format(self.url)
        else:
            return FM_GET_SPECS_BETA.format(self.url)  # Remove on CPD 5.0 release

    def get_fm_custom_foundation_models_href(self) -> str:
        return FM_GET_CUSTOM_FOUNDATION_MODELS.format(self.url)

    def get_fm_tasks_href(self, limit: str) -> str:
        return FM_GET_TASKS.format(self.url, limit)

    def get_fm_chat_href(self, item: str) -> str:
        return FM_CHAT.format(self.url, item)

    def get_fm_generation_href(self, item: str | None = None) -> str:
        if self._use_fm_ga_api:
            return FM_GENERATION.format(self.url)
        else:
            return FM_GENERATION_BETA.format(
                self.url, item
            )  # Remove on CPD 5.0 release

    def get_fm_generation_stream_href(self) -> str:
        return FM_GENERATION_STREAM.format(self.url)

    def get_fm_tokenize_href(self) -> str:
        if self._use_fm_ga_api:
            return FM_TOKENIZE.format(self.url)
        else:
            return FM_TOKENIZE_BETA.format(self.url)  # Remove on CPD 5.0 release

    def get_fm_deployment_generation_href(
        self, deployment_id: str, item: str | None = None
    ) -> str:
        if self._use_fm_ga_api:
            return FM_DEPLOYMENT_GENERATION.format(self.url, deployment_id)
        else:
            return FM_DEPLOYMENT_GENERATION_BETA.format(
                self.url, deployment_id, item
            )  # Remove on CPD 5.0 release

    def get_fm_deployment_generation_stream_href(self, deployment_id: str) -> str:
        return FM_DEPLOYMENT_GENERATION_STREAM.format(self.url, deployment_id)

    def get_fm_deployment_chat_href(self, deployment_id: str) -> str:
        return FM_DEPLOYMENT_CHAT.format(self.url, deployment_id)

    def get_fm_deployment_chat_stream_href(self, deployment_id: str) -> str:
        return FM_DEPLOYMENT_CHAT_STREAM.format(self.url, deployment_id)

    def get_ai_services_deployment_generation_href(
        self, deployment_id: str, item: str | None = None
    ) -> str:
        return AI_SERVICES_DEPLOYMENT_GENERATION.format(self.url, deployment_id)

    def get_ai_services_deployment_generation_stream_href(
        self, deployment_id: str, item: str | None = None
    ) -> str:
        return AI_SERVICES_DEPLOYMENT_GENERATION_STREAM.format(self.url, deployment_id)

    def get_prompts_href(self, ga_api: bool = True) -> str:
        return (GA_PROMPTS if ga_api else PROMPTS).format(
            self._get_platform_url_if_exists()
        )

    def get_text_detection_href(self) -> str:
        return TEXT_DETECTION.format(self.url)

    def get_text_extractions_href(self) -> str:
        return TEXT_EXTRACTIONS.format(self.url)

    def get_text_extraction_href(self, text_extraction_id: str) -> str:
        return TEXT_EXTRACTION.format(self.url, text_extraction_id)

    def get_rerank_href(self) -> str:
        return RERANK.format(self.url)

    def get_prompts_all_href(self) -> str:
        return PROMPTS_GET_ALL.format(self._get_platform_url_if_exists())

    def get_parameter_set_href(self, parameter_sets_id: str) -> str:
        return PARAMETER_SET.format(
            self._get_platform_url_if_exists(), parameter_sets_id
        )

    def get_parameter_sets_href(self) -> str:
        return PARAMETER_SETS.format(self._get_platform_url_if_exists())

    def get_runtime_definition_href(self, runtime_definitions_id: str) -> str:
        return RUNTIME_DEFINITION.format(
            self._get_platform_url_if_exists(), runtime_definitions_id
        )

    def get_runtime_definitions_href(self) -> str:
        return RUNTIME_DEFINITIONS.format(self._get_platform_url_if_exists())

    def get_fm_embeddings_href(self):
        return FM_EMBEDDINGS.format(self.url)

    def get_fine_tuning_href(self, tuning_id: str):
        return FM_FINE_TUNING.format(self.url, tuning_id)

    def get_fine_tunings_href(self):
        return FM_FINE_TUNINGS.format(self.url)

    def get_autoai_rag_href(self):
        return AUTOAI_RAG.format(self.url)

    def get_autoai_rag_id_href(self, rag_id):
        return AUTOAI_RAG_ID.format(self.url, rag_id)

    def get_time_series_href(self) -> str:
        return FM_TIME_SERIES.format(self.url)

    def get_audio_transcriptions_href(self) -> str:
        return FM_AUDIO_TRANSCRIPTIONS.format(self.url)

    def get_deployment_time_series_href(self, deployment_id: str) -> str:
        return FM_DEPLOYMENT_TIME_SERIES.format(self.url, deployment_id)

    def get_taxonomy_href(self, taxonomy_id: str) -> str:
        return TAXONOMY.format(self.url, taxonomy_id)

    def get_taxonomies_imports_href(self) -> str:
        return TAXONOMIES_IMPORTS.format(self.url)

    def get_taxonomies_import_href(self, taxonomy_import_id: str):
        return TAXONOMIES_IMPORT.format(self.url, taxonomy_import_id)

    def get_document_extractions_href(self):
        return DOCUMENT_EXTRACTIONS.format(self.url)

    def get_document_extraction_href(self, document_extraction_id: str):
        return DOCUMENT_EXTRACTION.format(self.url, document_extraction_id)

    def get_synthetic_data_generations_href(self):
        return SYNTHETIC_DATA_GENERATIONS.format(self.url)

    def get_synthetic_data_generation_href(self, sdg_id: str):
        return SYNTHETIC_DATA_GENERATION.format(self.url, sdg_id)

    def get_utility_agent_tools_href(self):
        return UTILITY_AGENT_TOOLS_BETA.format(self._get_platform_url_if_exists())

    def get_utility_agent_tools_run_href(self):
        return UTILITY_AGENT_TOOLS_RUN_BETA.format(self._get_platform_url_if_exists())

    def get_jobs_runs_href(self, job_id: str, run_id: str):
        return JOBS_RUNS.format(self._get_platform_url_if_exists(), job_id, run_id)

    def get_vector_indexes_href(self):
        return VECTOR_INDEXES.format(self._get_platform_url_if_exists())

    def get_vector_index_href(self, vector_index_id: str):
        return VECTOR_INDEX.format(self._get_platform_url_if_exists(), vector_index_id)

    def get_vector_indexes_all_href(self):
        return VECTOR_INDEXES_GET_ALL.format(self._get_platform_url_if_exists())

    def get_gateway_tenant_href(self):
        return GATEWAY_TENANT.format(self.url)

    def get_gateway_providers_href(self):
        return GATEWAY_PROVIDERS.format(self.url)

    def get_gateway_provider_href(self, provider_id):
        return GATEWAY_PROVIDER.format(self.url, provider_id)

    def get_gateway_provider_available_models_href(self, provider_id):
        return GATEWAY_PROVIDER_AVAILABLE_MODELS.format(self.url, provider_id)

    def get_gateway_update_provider_href(self, provider_id, provider):
        return GATEWAY_UPDATE_PROVIDER.format(self.url, provider_id, provider)

    def get_gateway_models_href(self, provider_id):
        return GATEWAY_MODELS.format(self.url, provider_id)

    def get_gateway_all_tenant_models_href(self):
        return GATEWAY_ALL_TENANT_MODELS.format(self.url)

    def get_gateway_model_href(self, model_id: str) -> str:
        return GATEWAY_MODEL.format(self.url, model_id)

    def get_gateway_policies_href(self):
        return GATEWAY_POLICIES.format(self.url)

    def get_gateway_embeddings_href(self):
        return GATEWAY_EMBEDDINGS.format(self.url)

    def get_gateway_text_completions_href(self):
        return GATEWAY_TEXT_COMPLETIONS.format(self.url)

    def get_gateway_chat_completions_href(self):
        return GATEWAY_CHAT_COMPLETIONS.format(self.url)
