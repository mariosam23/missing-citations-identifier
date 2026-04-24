import logging
import sys

logger = logging.getLogger("missing_citations")

def configure(level: int = logging.INFO) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

# Configure logger on import so other modules get a working logger quickly.
configure()
