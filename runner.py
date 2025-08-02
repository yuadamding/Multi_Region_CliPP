import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configures the logging for the script to report on a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler for real-time info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    # Rotating file handler for detailed log report
    file_handler = RotatingFileHandler('subdir_processing.log', maxBytes=2*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# Initialize the logger
logger = setup_logging()

# Import the necessary library from your project
try:
    from clipp2.core import *
except ImportError:
    logger.warning("Could not import 'clipp2.core'. Please ensure the library is installed.")

def run_script_on_subdirectories(root_directory='input'):
    """
    Finds immediate subdirectories in a root directory and executes a script
    for each one, passing the subdirectory path as an argument.

    Args:
        root_directory (str): The path to the directory containing the subdirectories.
    """
    script_to_run = 'clipp2_model_selection_2.py'
    logger.info(f"Starting process for root directory: '{root_directory}'")

    if not os.path.isdir(root_directory):
        logger.critical(f"Error: The root directory '{root_directory}' does not exist.")
        return

    if not os.path.isfile(script_to_run):
        logger.critical(f"Error: The target script '{script_to_run}' was not found.")
        return

    # Get all items in the root directory
    try:
        sub_items = os.listdir(root_directory)
    except OSError as e:
        logger.critical(f"Could not read directory '{root_directory}': {e}")
        return

    # Loop through each item and check if it is a directory
    for item_name in sub_items:
        # Construct the full path for the item
        subdir_path = os.path.join(root_directory, item_name)

        if os.path.isdir(subdir_path):
            logger.info(f"--- Found subdirectory. Processing: {subdir_path} ---")

            # Construct the command with the subdirectory path as the input
            command = ["python", script_to_run,'-i', subdir_path, '-o', 'output1']
            logger.debug(f"Executing command: {' '.join(command)}")

            try:
                # Execute the command
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Successfully executed script for '{subdir_path}'")
                if result.stdout:
                    # Log the output from the script for the report
                    logger.debug(f"--- Begin STDOUT for {subdir_path} ---\n{result.stdout.strip()}\n--- End STDOUT ---")

            except subprocess.CalledProcessError as e:
                # Log an error if the script fails
                logger.error(f"Error running script for '{subdir_path}'. Return Code: {e.returncode}")
                if e.stderr:
                    logger.error(f"--- Begin STDERR for {subdir_path} ---\n{e.stderr.strip()}\n--- End STDERR ---")
            except Exception as e:
                logger.critical(f"An unexpected error occurred while processing '{subdir_path}': {e}", exc_info=True)
        else:
            logger.debug(f"Skipping '{item_name}' because it is not a directory.")

    logger.info("Processing of all subdirectories finished.")

if __name__ == "__main__":
    # The script will run on the subdirectories inside 'input' when executed.
    run_script_on_subdirectories('input1')