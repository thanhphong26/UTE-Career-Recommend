import logging
import os
import sys
from config import Config

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(Config.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Create formatter
formatter = logging.Formatter(Config.LOG_FORMAT)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))

# Clear existing handlers (to avoid duplicates)
if root_logger.handlers:
    root_logger.handlers.clear()

# File handler with UTF-8 encoding
file_handler = logging.FileHandler(Config.LOG_FILE, encoding=Config.LOG_ENCODING)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Special handling for StreamHandler with proper encoding
# Use NullHandler if in Windows environment to avoid console encoding issues
if os.name == 'nt':  # Windows system
    # Option 1: Disable console logging completely to avoid encoding errors
    # null_handler = logging.NullHandler()
    # root_logger.addHandler(null_handler)
    
    # Option 2: Use sys.stdout with UTF-8 encoding (safer option)
    # Force stdout to use UTF-8 encoding
    sys.stdout.reconfigure(encoding=Config.LOG_ENCODING, errors='backslashreplace')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
else:
    # On Unix systems, StreamHandler works fine with UTF-8
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)

# For backward compatibility
logger = logging.getLogger(__name__)