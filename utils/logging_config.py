from loguru import logger

logger.remove()

logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation = "1 week",
    retention = "1 month",
    compression = "zip",
    level = "DEBUG"
)

__all__ = ["logger"]