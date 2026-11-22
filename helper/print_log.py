"""
Backward-compatible logging helpers for Clipty-Mini.
Routes legacy info/error/sucess calls to standard Python logging.
"""

from helper.logger import logger


def info(text: str):
    logger.info(text)


def error(text: str):
    logger.error(text)


def sucess(text: str):
    logger.info(f"SUCCESS: {text}")
