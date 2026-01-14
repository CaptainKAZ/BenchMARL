"""Debug utilities for model development"""
import logging
import inspect
import torch
import os
from pathlib import Path
from typing import Any, Optional

# Configure logger
logger = logging.getLogger("benchmarl.models")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # 防止重复输出

# 默认格式化器
formatter = logging.Formatter(
    '%(levelname)s - %(message)s'
)


def setup_model_logging(
    log_to_console: bool = True,
    log_to_file: bool = False,
    log_file_path: Optional[str] = None,
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    clear_existing_handlers: bool = True,
):
    """
    配置模型调试日志的输出方式

    Args:
        log_to_console: 是否输出到控制台，默认 True
        log_to_file: 是否输出到文件，默认 False
        log_file_path: 日志文件路径，如果为 None 则使用 './model_debug.log'
        console_level: 控制台日志级别 (DEBUG/INFO/WARNING/ERROR)
        file_level: 文件日志级别 (DEBUG/INFO/WARNING/ERROR)
        clear_existing_handlers: 是否清除已有的 handlers，默认 True

    Example:
        # 只输出到控制台 (默认)
        setup_model_logging()

        # 只输出到文件
        setup_model_logging(log_to_console=False, log_to_file=True,
                           log_file_path="outputs/model_debug.log")

        # 同时输出到控制台和文件
        setup_model_logging(log_to_console=True, log_to_file=True,
                           log_file_path="outputs/model_debug.log")

        # 控制台只显示 WARNING，文件记录所有 DEBUG
        setup_model_logging(log_to_console=True, log_to_file=True,
                           console_level="WARNING", file_level="DEBUG",
                           log_file_path="outputs/model_debug.log")
    """
    global logger

    # 清除已有的 handlers
    if clear_existing_handlers:
        logger.handlers.clear()

    # 添加控制台 handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        print(f"✓ Model debug logs enabled for console (level: {console_level})")

    # 添加文件 handler
    if log_to_file:
        if log_file_path is None:
            log_file_path = "./model_debug.log"

        # 创建目录（如果不存在）
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))

        # 文件使用更详细的格式
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        print(f"✓ Model debug logs enabled for file: {log_file_path} (level: {file_level})")

    if not log_to_console and not log_to_file:
        # 禁用所有输出
        logger.addHandler(logging.NullHandler())
        print("⚠ Model debug logs disabled")


# 初始化默认配置（可以通过环境变量控制）
_default_console = os.getenv("MODEL_DEBUG_CONSOLE", "false").lower() == "true"
_default_file = os.getenv("MODEL_DEBUG_FILE", "false").lower() == "true"
_default_file_path = os.getenv("MODEL_DEBUG_FILE_PATH", None)
_default_console_level = os.getenv("MODEL_DEBUG_CONSOLE_LEVEL", "DEBUG")
_default_file_level = os.getenv("MODEL_DEBUG_FILE_LEVEL", "DEBUG")

# 自动初始化
if not logger.handlers:
    setup_model_logging(
        log_to_console=_default_console,
        log_to_file=_default_file,
        log_file_path=_default_file_path,
        console_level=_default_console_level,
        file_level=_default_file_level,
        clear_existing_handlers=False,
    )


def debug_print(model_name: str, description: str, tensor: Any, extra_info: str = ""):
    """
    Print debug information with automatic file/line number detection

    Args:
        model_name: Name of the model (e.g., self.name)
        description: Description of what's being printed
        tensor: The tensor or value to print shape/info for
        extra_info: Additional information to display
    """
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split('/')[-1]  # Just the filename
    lineno = frame.f_lineno

    # Format tensor info
    if isinstance(tensor, torch.Tensor):
        tensor_info = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    elif isinstance(tensor, (list, tuple)):
        tensor_info = f"len={len(tensor)}, type={type(tensor).__name__}"
    else:
        tensor_info = f"value={tensor}, type={type(tensor).__name__}"

    # Build message
    location = f"[{filename}:{lineno}]"
    model_info = f"[{model_name}]"
    extra = f" | {extra_info}" if extra_info else ""

    message = f"{location} {model_info} {description}: {tensor_info}{extra}"
    logger.debug(message)


def debug_separator(model_name: str, section: str):
    """Print a separator for readability"""
    logger.debug(f"{'='*80}")
    logger.debug(f"[{model_name}] {section}")
    logger.debug(f"{'='*80}")
