# -*- coding: utf-8 -*-
"""通用日志模块：统一配置标准库 logging，供各模块复用。

用法：
    from component.logger import get_logger, setup_logging
    setup_logging(level="INFO", path="logs/app.log")   # 应用启动时按配置初始化
    log = get_logger("context_manager")
    log.info("...")
"""
import logging
import os
import sys

_ROOT_LOGGER = "ai_demo"
_CONFIGURED = False
_FILE_HANDLERS = set()   # 已添加的文件路径，避免重复 handler

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure() -> None:
    """幂等配置：仅初始化控制台 handler 一次。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


def setup_logging(level: str = "INFO", path: str = "") -> None:
    """按配置设置日志级别与输出：控制台 + 可选文件。

    :param level: DEBUG | INFO | WARNING | ERROR（大小写不敏感）
    :param path: 日志文件路径；为空则仅输出到控制台
    """
    _configure()
    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if path:
        if path not in _FILE_HANDLERS:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            root.addHandler(fh)
            _FILE_HANDLERS.add(path)


def get_logger(name: str) -> logging.Logger:
    """获取统一命名空间的 logger（如 get_logger('context_manager')）。"""
    _configure()
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")
