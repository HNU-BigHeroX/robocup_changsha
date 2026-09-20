"""错误类型体系：CoverageError 基类携带稳定错误码，各子类对应固定默认错误码。"""


class CoverageError(Exception):
    """评测库统一错误基类，携带 message、稳定错误码 code 与可选 field。"""
    def __init__(self, message: str, code: str = "COVERAGE_ERROR", field: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.field = field


class ConfigurationError(CoverageError):
    def __init__(self, message: str, code: str = "CONFIG_INVALID", field: str | None = None):
        super().__init__(message, code=code, field=field)


class VersionMismatchError(CoverageError):
    def __init__(self, message: str, code: str = "VERSION_MISMATCH", field: str | None = None):
        super().__init__(message, code=code, field=field)


class ActionValidationError(CoverageError):
    def __init__(self, message: str, code: str = "ACTION_INVALID", field: str | None = "action"):
        super().__init__(message, code=code, field=field)


class ObservationValidationError(CoverageError):
    def __init__(self, message: str, code: str = "OBSERVATION_INVALID", field: str | None = None):
        super().__init__(message, code=code, field=field)


class UnknownAgentError(CoverageError):
    def __init__(self, message: str, code: str = "UNKNOWN_AGENT", field: str | None = None):
        super().__init__(message, code=code, field=field)


class EnvironmentStateError(CoverageError):
    def __init__(self, message: str, code: str = "ENV_STATE_INVALID", field: str | None = None):
        super().__init__(message, code=code, field=field)


class ScenarioGenerationError(CoverageError):
    def __init__(self, message: str, code: str = "SCENARIO_GENERATION_FAILED", field: str | None = None):
        super().__init__(message, code=code, field=field)
