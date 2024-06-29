import os
import yaml
import logging
import json
import readchar
import glob
from colorama import Fore, Style, init
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml file."""
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def create_session_folder():
    """Create a new folder for the current session."""
    session_folder = f"Collections/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def save_transcription(folder, original, translated):
    """Save transcription and translation to a file."""
    with open(os.path.join(folder, "transcriptions.txt"), "a", encoding="utf-8") as file:
        file.write(f"Original: {original}\nTranslated: {translated}\n\n")

def save_to_desktop(file_name, content):
    """Save content to a file on the desktop."""
    desktop_path = Path.home() / "Desktop"
    file_path = desktop_path / file_name
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Saved: {file_path}")

from encoding_utils import safe_print
    
def print_json_formatted(data, indent=4, width_percentage=0.65):
    """
    Prints a dictionary in a formatted JSON style within the terminal, offering a visually structured representation of the data.
    This function enhances readability by formatting the JSON data with indentation and color-coding, and adapts to the width of the terminal.

    It's particularly useful for displaying complex data structures in a human-readable format, which can be beneficial for debugging,
    logging, or presenting information to the user.

    Args:
        data (dict): The dictionary containing the data to be printed.
        indent (int, optional): The number of spaces used for indentation in the JSON output. Defaults to 4.
        width_percentage (float, optional): The percentage of the terminal's width to be used for text wrapping.
                                            This helps in ensuring the JSON data fits within the terminal window. Defaults to 0.65.

    Returns:
        None

    Notes:
        - The function uses 'json.dumps' for converting dictionary keys and values into a JSON-formatted string.
        - The terminal's width is dynamically calculated to adapt the output formatting to different terminal sizes.
        - Color coding is applied for different elements: keys are highlighted in yellow, and values are colored
          based on their types (e.g., green for transcriptions, magenta for translations).
        - The function is designed to handle and properly display nested dictionaries and lists.

    Example:
        data = {'Transcription': 'Hello, world!', 'Translation': 'Hola, mundo!'}
        print_json_formatted(data)
        # Output will be a color-coded, indented JSON representation of the data dictionary.
    """
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    max_width = int(terminal_width * width_percentage)

    for key, value in data.items():
        key_str = json.dumps({key: ""}, indent=indent, ensure_ascii=False).rstrip(": {}\n")
        value_str = json.dumps(value, ensure_ascii=False)
        wrapper = textwrap.TextWrapper(
            width=max_width, subsequent_indent=" " * (indent + len(key) + 4)
        )
        if key == "Transcription":
            color = Fore.GREEN
        elif key == "Translation":
            color = Fore.MAGENTA
        else:
            color = Fore.CYAN
        safe_print(wrapper.fill(Fore.YELLOW + key_str + Style.RESET_ALL + ":"))
        safe_print(color + value_str + Style.RESET_ALL)


# Add more utility functions as needed