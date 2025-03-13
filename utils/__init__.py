import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="Log: %(message)s",  # Show only the log message
)

# Create a logger instance
logger = logging.getLogger(__name__)