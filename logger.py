import logging
import sys

# Create a custom logger
logger = logging.getLogger("chatkgp")
logger.setLevel(logging.DEBUG)  # Set lowest level to capture all logs

# Create handler for stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # You can adjust this if needed

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    COLORS = {
        "DEBUG": "\033[38;5;247m",     # Grey
        "INFO": "\033[0m",      # Default
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"
    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.RESET)
        log_fmt = f"[%(asctime)s] — ({level_color}%(levelname)s{self.RESET}) — {level_color}%(message)s{self.RESET}"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Formatter
formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Avoid duplicate logs if handler already added
if not logger.hasHandlers():
    logger.addHandler(console_handler)
