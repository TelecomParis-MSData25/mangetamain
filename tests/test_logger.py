import logging

from src import logger as logger_module


def test_configure_logger_creates_handlers(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logger_module, "LOG_DIR", str(log_dir))
    monkeypatch.setattr(logger_module, "DEBUG_LOG_FILE", str(log_dir / "debug.log"))
    monkeypatch.setattr(logger_module, "ERROR_LOG_FILE", str(log_dir / "errors.log"))

    configured_logger = logger_module._configure_logger()
    handler_levels = {handler.level for handler in configured_logger.handlers}

    assert logging.DEBUG in handler_levels
    assert logging.ERROR in handler_levels
    assert log_dir.exists()

    for handler in configured_logger.handlers:
        handler.close()


def test_disable_enable_logging():
    logger_module.enable_logging()
    logger_module.disable_logging()
    assert logging.getLogger("webapp").level == logging.CRITICAL

    logger_module.enable_logging()
    assert logging.getLogger("webapp").level == logging.DEBUG
