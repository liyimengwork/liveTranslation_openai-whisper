import readchar
from colorama import Fore, Style

def print_welcome_message():
    print(Fore.GREEN + "\nWelcome to the real-time translation tool.\n" + Style.RESET_ALL)

def get_language_choice(language_map):
    """
    Prompts the user to select a language from a given language map and returns the selected language.

    Parameters:
        language_map (dict): A dictionary containing the available languages as keys.

    Returns:
        str: The selected language.

    Raises:
        ValueError: If the user enters an invalid choice.
    """
    print("Please select a language:")
    for i, lang in enumerate(language_map.keys(), 1):
        print(f"{i}. {lang}")
    choice = int(input("Enter the number of your choice: ")) - 1
    return list(language_map.keys())[choice]

def get_file_processing_choices():
    """
    Prompts the user to choose an action and a file or directory path.

    Returns:
        tuple: A tuple containing the user's action choice (1 or 2) and the file or directory path.
        Returns (None, None) if the user enters an invalid choice.
    """
    action_choice = input(
        "Choose action: 1 for 'transcribe and translate', 2 for 'only transcribe': "
    ).strip()
    if action_choice not in ["1", "2"]:
        print("Invalid choice. Exiting.")
        return None, None

    file_choice = input("Process all audio files in the same directory? (yes/no): ").strip().lower()
    if file_choice == "yes":
        directory_path = input("Enter the directory path: ").strip()
        return action_choice, directory_path
    elif file_choice == "no":
        file_path = input("Enter the full path to the audio file: ").strip()
        return action_choice, file_path
    else:
        print("Invalid choice. Exiting.")
        return None, None

def single_run_input_loop():
    """
    This function displays a prompt to the user asking them to press the space bar to start recording, 'r' to replay the last translation, or 'exit' to quit. It then reads a single character from the user's input and returns it.

    Returns:
        str: The character that the user pressed.
    """
    print(
        Fore.GREEN
        + "Press the space bar to start recording, 'r' to replay the last translation, or 'exit' to quit:"
        + Style.RESET_ALL
    )
    return readchar.readkey()

# Add more CLI-related functions as needed