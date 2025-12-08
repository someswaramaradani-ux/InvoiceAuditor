# logger_config.py
import logging
import os

LOG_FILE = "invoice_auditor.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger():
    """Configures the root logger for the entire application."""
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'), # Log to a file
            logging.StreamHandler()                  # Log to console
        ]
    )
    # Return a specific logger instance for use in agents
    return logging.getLogger("InvoiceAuditor")

# Initialize and get the logger instance
AUDITOR_LOGGER = setup_logger()