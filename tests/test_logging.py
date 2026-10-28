"""Exercise real vLLM logging in fresh processes, without loading an NPU model."""

import json
import os
import subprocess
import sys

import pytest

MODULES = (
    "vllm_mblt.mblt_platform",
    "vllm_mblt.mblt_worker",
    "vllm_mblt.host_threads",
    "vllm_mblt.tracing",
)


def _run(code, **env):
    process_env = os.environ.copy()
    for key in ("VLLM_LOGGING_CONFIG_PATH", "VLLM_LOGGING_LEVEL", "VLLM_CONFIGURE_LOGGING"):
        process_env.pop(key, None)
    process_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + code],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout


# Serving applies LoggingConfig explicitly; importing vllm.logger alone does not configure its handlers.
BOOTSTRAP = """
from vllm.config.logging import LoggingConfig
from vllm.logger import configure_logging
configure_logging(LoggingConfig())
"""

EMIT = f"""
import importlib
for name in {MODULES!r}:
    module = importlib.import_module(name)
    importlib.reload(module)
    assert module.logger.name == "vllm." + name
    assert not module.logger.handlers
    module.logger.info("PLUGIN_INFO_" + name)
    module.logger.warning("PLUGIN_WARNING_" + name)
"""


def test_default_logging_emits_each_plugin_message_once():
    output = _run(EMIT)
    for module in MODULES:
        assert output.count("PLUGIN_INFO_" + module) == 1
        assert output.count("PLUGIN_WARNING_" + module) == 1


def test_warning_level_suppresses_plugin_info():
    output = _run(EMIT, VLLM_LOGGING_LEVEL="WARNING")
    assert "PLUGIN_INFO_" not in output
    for module in MODULES:
        assert output.count("PLUGIN_WARNING_" + module) == 1


def test_disabled_logging_does_not_install_handlers():
    _run(
        f"""
import importlib
import logging
for name in {MODULES!r}:
    module = importlib.import_module(name)
    assert not module.logger.handlers
assert not logging.getLogger("vllm").handlers
assert not logging.getLogger().handlers
""",
        VLLM_CONFIGURE_LOGGING="0",
    )


@pytest.mark.parametrize("configure_after_import", [False, True])
def test_custom_logging_config_is_respected(tmp_path, configure_after_import):
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"custom": {"format": "CUSTOM %(name)s %(message)s"}},
        "handlers": {"custom": {"class": "logging.StreamHandler", "formatter": "custom", "stream": "ext://sys.stdout"}},
        "loggers": {"vllm": {"handlers": ["custom"], "level": "INFO", "propagate": False}},
    }
    path = tmp_path / "logging.json"
    path.write_text(json.dumps(config))
    code = EMIT
    if configure_after_import:
        code = (
            f"""
import importlib
import json
import logging.config
for name in {MODULES!r}:
    importlib.import_module(name)
with open({str(path)!r}) as f:
    logging.config.dictConfig(json.load(f))
"""
            + EMIT
        )
    output = _run(code, **({} if configure_after_import else {"VLLM_LOGGING_CONFIG_PATH": str(path)}))
    for module in MODULES:
        assert output.count("CUSTOM vllm." + module + " PLUGIN_INFO_" + module) == 1
