# -*- coding: utf-8 -*-
"""通用日志模块：统一配置标准库 logging，供各模块复用。

用法：
    from component.logger import get_logger
    log = get_logger("context_manager")
    log.info("...")
"""
import logging
import sys

_ROOT_LOGGER = "ai_demo"
_CONFIGURED = False


def _configure() -> None:
    """幂等配置：仅初始化一次，避免重复添加 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """获取统一命名空间的 logger（如 get_logger('context_manager')）。"""
    _configure()
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")
