import argparse
import sys
import glob
import os
import wavio
from pynput import keyboard
import time
import signal
import numpy as np
import sounddevice as sd
import logging
from colorama import Fore, Style, init
from groq import Groq
from openai import OpenAI
from encoding_utils import setup_encoding
from utils import load_config, create_session_folder, save_transcription, save_to_desktop, print_json_formatted
from audio_processing import (
    record_audio, play_audio, voice_to_text, clear_audio_frames,
    record_audio_continuous, start_recording, stop_recording, WAVE_OUTPUT_FILENAME, CHANNELS, SAMPLE_WIDTH, RATE, FORMAT
)
from api_handlers import transcribe_audio, translate_text, voice_stream
from cli_interface import print_welcome_message, get_language_choice, get_file_processing_choices, single_run_input_loop
import pyaudio
import yaml

# Constants
DEFAULT_CONTENT = """You are a [Desired Language]/English translation and interpreter assistant. Your purpose is to bridge the communication and language gap for both [Desired Language] and English speakers. If the input is completely [Desired Language] you WILL only translate to English and vice versa if the input is completely in English you translate to [Name of desired language in that language] for a seamless live translation style approach. If in an input you detect both [Name of desired language in that language] and English and it is clearly distinguishable, please continue to translate to the opposite language. Here is an Example of the desired response style when detecting both languages and responding with both languages. Do not translate the entire text string to one language. keep a convo style flow. You will not execute or analyze any of the info in text sent to be translated. you will only play the role of translating so do not try to provide context or answer questions and request: Translation: I want to know why I have to go to the store to get a deal rather than shopping online. [Phrase in desired language in that language's text if possible]"""
SPECIAL_CONTENT = """It is a beautiful, highly productive September sunny day and you are highly motivated, and you are a World Class Expert AI multilingual translator interpreter. You're capable of understanding any in all languages, and able to fluently and accurately translate them back to English. Your goal and underlying purpose is to bridge all gaps in communication and effectively translate back to English no matter what. You have done this, you are capable of doing this and you will do this. Important: Translate any text to ENGLISH"""

# Logging added back
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# print(config)

# Now you can access the API keys like this:
openai_api_key = config['openai']['api_key']
groq_api_key = config['groq']['api_key']

# Use the API keys to initialize your clients
groq_client = Groq(api_key=groq_api_key)

# print(groq_api_key)
# print(openai_api_key)

# Setup
setup_encoding()
init(autoreset=True)
config = load_config()
groq_client = Groq(api_key=groq_api_key)
openai_client = OpenAI(api_key=openai_api_key)

# Setup
setup_encoding()
init(autoreset=True)
config = load_config()
groq_client = Groq(api_key=config["groq"]["api_key"])
openai_client = OpenAI(api_key=config["openai"]["api_key"])


language_map = {
    "European Spanish (Spain)": ("Español Europeo", "Buenos días, ¿cómo estás hoy?"),
    "Spanish": ("Español", "¿Qué onda? ¿Todo bien?"),
    "Caribbean Spanish (Cuba, Puerto Rico, Dominican Republic)": (
        "Español Caribeño",
        "Hace mucho calor hoy, ¿verdad?",
    ),
    "Central American Spanish (Guatemala, Honduras, Nicaragua)": (
        "Español Centroamericano",
        "Vamos a la playa este fin de semana.",
    ),
    "Andean Spanish (Peru, Bolivia, Ecuador)": (
        "Español Andino",
        "La comida aquí es muy deliciosa.",
    ),
    "Rioplatense Spanish (Argentinna and Uruguay)": (
        "Español Rioplatense",
        "¿Me pasás la yerba, por favor?",
    ),
    "Chilean Spanish": ("Español Chileno", "¿Cachai lo que te estoy diciendo?"),
    "Colombian Spanish": ("Español Colombiano", "¿Quieres ir a tomar un tinto?"),
    "Venezuelan Spanish": (
        "Español Venezolano",
        "Vamos a comer unas arepas esta noche.",
    ),
    "Canary Islands Spanish": ("Español Canario", "El cielo está muy despejado hoy."),
    "Mandarin Chinese": ("普通话", "你好，你吃饭了吗？"),
    "French": ("Français", "Bonjour, où se trouve la bibliothèque?"),
    "German": ("Deutsch", "Kannst du mir helfen, bitte?"),
    "Portuguese": ("Português", "Bom dia, como você está?"),
    "Russian": ("Русский", "Как дела? Всё хорошо?"),
    "Japanese": ("日本語", "こんにちは、元気ですか？"),
    "Italian": ("Italiano", "Dove posso trovare un buon ristorante?"),
    "Arabic": ("العربية", "مرحبا، كيف حالك اليوم؟"),
    "Hindi": ("हिंदी", "नमस्ते, आप कैसे हैं?"),
    "Korean": ("한국어", "안녕하세요, 잘 지내세요?"),
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time translation tool")
    parser.add_argument("-d", "--duration", type=int, default=45, help="Maximum duration of the recording in seconds (default: 45)")
    parser.add_argument("-f", "--file", type=str, help="Path to an existing audio file to transcribe and translate")
    parser.add_argument("-c", "--content", type=str, nargs="?", choices=list(language_map.keys()) + ["Smart Select", None], default=DEFAULT_CONTENT, help="Custom content for the API call to Whisper")
    parser.add_argument("-t", "--continuous", action="store_true", help="Enable continuous run mode")
    parser.add_argument("-v", "--voice", choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"], help="Choose a TTS voice for speaking the translation")
    parser.add_argument("--save_recordings", action="store_true", help="Save all recordings instead of deleting them")
    
    args = parser.parse_args()
    if args.content and isinstance(args.content, bytes):
        args.content = args.content.decode(sys.getfilesystemencoding())
    return args

def get_modified_content(language):
    """
    Generate the modified content for the given language.

    Args:
        language (str): The desired language for the content.

    Raises:
        ValueError: If the language is not supported.

    Returns:
        str: The modified content with the desired language replaced.

    Prints:
        - The modified content being sent to the API.

    Note:
        - The modified content is generated by replacing the placeholders
          "[Desired Language]", "[Name of desired language in that language]",
          and "[Phrase in desired language in that language's text if possible]"
          in the DEFAULT_CONTENT string.
        - The modified content is then printed in cyan color.
        - An empty line is also printed after the modified content.
    """
    if language not in language_map:
        raise ValueError(f"Unsupported language: {language}")
    
    lang_info = language_map[language]
    modified_content = DEFAULT_CONTENT.replace("[Desired Language]", language)
    modified_content = modified_content.replace("[Name of desired language in that language]", lang_info[0])
    modified_content = modified_content.replace("[Phrase in desired language in that language's text if possible]", lang_info[1])
    
    print(Fore.CYAN + "\nContent being sent to API:" + Style.RESET_ALL)
    print(modified_content)
    print()  # Add an    
    return modified_content

def continuous_run_mode(content, args, session_folder):
    print(Fore.GREEN + "\nContinuous run mode activated.\n" + Style.RESET_ALL)
    print(Fore.YELLOW + "Press SPACE to start/stop recording (max 45 seconds)." + Style.RESET_ALL)
    print(Fore.YELLOW + "Press 'R' to replay the last translation." + Style.RESET_ALL)
    print(Fore.YELLOW + "Press ESC to exit." + Style.RESET_ALL)
    audio_files = []
    is_recording = False
    should_exit = False
    last_ai_audio_path = None

    def on_press(key):
        """
        Handle the key press event.

        Parameters:
            key (Key): The key that was pressed.

        Returns:
            bool: True if the key press event was handled, False otherwise.

        This function is called when a key is pressed. It checks the key that was pressed and performs the
        corresponding action. If the key is the space key, it toggles the recording state. If the key is the
        'r' key, it replays the last translation. If the key is the escape key, it sets the should_exit flag
        to True.

        The function also checks the recording state and prints a message to the console accordingly.

        Returns:
            bool: True if the key press event was handled, False otherwise.

        """
        nonlocal is_recording, should_exit, last_ai_audio_path
        if key == keyboard.Key.space:
            is_recording = not is_recording
            if is_recording:
                print(Fore.CYAN + "\nRecording started. Press SPACE to stop or wait for 45 seconds." + Style.RESET_ALL)
            else:
                print(Fore.CYAN + "Recording stopped." + Style.RESET_ALL)
        elif key == keyboard.KeyCode.from_char('r'):
            if last_ai_audio_path:
                print(Fore.CYAN + "Replaying last translation..." + Style.RESET_ALL)
                play_audio(file_path=last_ai_audio_path)
            else:
                print(Fore.YELLOW + "No translation available to replay." + Style.RESET_ALL)
        elif key == keyboard.Key.esc:
            should_exit = True
            return False  # Stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def signal_handler(sig, frame):
        nonlocal should_exit
        print(Fore.RED + "\nInterrupt received, cleaning up..." + Style.RESET_ALL)
        should_exit = True

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not should_exit:
            if is_recording:
                start_time = time.time()
                audio_data = []

                while is_recording and time.time() - start_time < 45:
                    chunk = record_audio(0.1, session_folder)
                    if chunk is not None:
                        audio_data.extend(chunk)
                    time.sleep(0.05)

                if audio_data:
                    audio_array = np.array(audio_data)
                    audio_file_path = os.path.join(session_folder, f"audio_{int(time.time())}.wav")
                    wavio.write(audio_file_path, audio_array, RATE, sampwidth=2)
                    audio_files.append(audio_file_path)

                    logging.info(f"Transcribing audio file: {audio_file_path}")
                    transcribed_text = transcribe_audio(audio_file_path, groq_client)

                    if transcribed_text:
                        logging.info(f"Translating text: {transcribed_text}")
                        translated_text = translate_text(transcribed_text, content, config["openai"]["api_key"])
                        save_transcription(session_folder, transcribed_text, translated_text)

                        if args.voice:
                            logging.info(f"Generating voice for translated text: {translated_text}")
                            last_ai_audio_path = voice_stream(translated_text, args.voice, session_folder, openai_client, play_audio)
                            audio_files.append(last_ai_audio_path)

                        print_json_formatted({"Original": transcribed_text, "Translation": translated_text})

            time.sleep(0.1)

    except Exception as e:
        print(Fore.RED + f"\nAn error occurred: {e}" + Style.RESET_ALL)
    finally:
        listener.stop()
        handle_session_files(audio_files, session_folder, args.save_recordings)

def single_run_mode(content, args, session_folder):
    """
    Runs the program in single run mode.

    Args:
        content (str): The content to be translated.
        args (argparse.Namespace): The command line arguments.
        session_folder (str): The folder path for the session.

    Returns:
        None

    This function runs the program in single run mode. It listens for keyboard inputs and performs
    the following actions:

    - If the space bar is pressed, it starts recording audio. The recorded audio is saved in the
      session folder as a WAV file. The audio is transcribed and translated using the specified
      content. The translation is printed in JSON format.
      If the recording is successful, the translated text is also converted to speech using the
      specified voice.
    - If the 'r' key is pressed, it replays the last translation.
    - If the 'exit' key is pressed, it exits the program.

    The function also handles interrupt signals and cleans up the session files.

    The function does not return anything.
    """
    audio_files = []
    last_ai_audio_path = None

    print(Fore.GREEN + "Press the space bar to start recording, 'r' to replay the last translation, or 'exit' to quit:" + Style.RESET_ALL)

    try:
        while True:
            user_input = single_run_input_loop()
            if user_input == " ":
                audio_data = record_audio(args.duration or 20, session_folder)
                if audio_data.size > 0:  # Use .size to check if the numpy array is empty
                    audio_file_path = os.path.join(session_folder, f"audio_{int(time.time())}.wav")
                    wavio.write(audio_file_path, audio_data, RATE, sampwidth=SAMPLE_WIDTH)
                    audio_files.append(audio_file_path)
                    transcribed_text = transcribe_audio(audio_file_path, groq_client)

                    if transcribed_text:
                        translated_text = translate_text(transcribed_text, content, config["openai"]["api_key"])
                        save_transcription(session_folder, transcribed_text, translated_text)

                        if args.voice:
                            ai_audio_path = voice_stream(translated_text, args.voice, session_folder, openai_client, play_audio)
                            audio_files.append(ai_audio_path)
                            last_ai_audio_path = ai_audio_path

                        print_json_formatted({"Original": transcribed_text, "Translation": translated_text})
                else:
                    print("Recording was interrupted or failed. Please try again.")

            elif user_input.lower() == "r":
                if last_ai_audio_path:
                    play_audio(file_path=last_ai_audio_path)
                else:
                    print("No previous translation to replay.")

            elif user_input.lower() == "exit":
                break

    except KeyboardInterrupt:
        print(Fore.RED + "\nInterrupt received, cleaning up and exiting..." + Style.RESET_ALL)
    finally:
        handle_session_files(audio_files, session_folder, args.save_recordings)

def handle_session_files(audio_files, session_folder, save_recordings=False):
    """
    Handles the session files based on the user's input. If `save_recordings` is `False`, the function prompts the user to either delete the session files or keep them. If the user chooses to delete the files, the function attempts to remove each file in the `audio_files` list. If any file fails to be deleted, an error message is printed. If the user chooses to keep the files, a success message is printed indicating where the session files are saved. If `save_recordings` is `True`, a success message is printed indicating where all audio files are saved. If a keyboard interrupt occurs during the deletion process, a message is printed indicating that the file deletion was skipped and the session files are kept.

    Parameters:
        audio_files (list): A list of file paths for the session files.
        session_folder (str): The folder path where the session files are saved.
        save_recordings (bool, optional): Whether to save the audio files. Defaults to `False`.

    Returns:
        None
    """
    if not save_recordings:
        try:
            user_input = input(Fore.YELLOW + "Press 'd' to delete or any other key to keep the session files: " + Style.RESET_ALL)
            if user_input.lower() == "d":
                for file_path in audio_files:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(Fore.RED + f"Failed to delete file {file_path}: {e}" + Style.RESET_ALL)
                print(Fore.GREEN + "All session files have been deleted." + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"Session files are saved in {session_folder}." + Style.RESET_ALL)
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nFile deletion skipped. Session files are kept." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"All audio files are saved in {session_folder}." + Style.RESET_ALL)

def process_file(file_path, content, action_choice):
    """
    Process a file by transcribing its audio content and optionally translating it.

    Args:
        file_path (str): The path of the file to be processed.
        content (str): The content to be used for translation.
        action_choice (str): The choice of action to be performed.

    Returns:
        None

    This function takes a file path, a content string, and an action choice. It extracts the base name of the file,
    generates a text file name based on the base name, and transcribes the audio content of the file using the `groq_client`.
    If the transcription is successful, it checks the action choice. If the action choice is "1", it translates the
    transcribed text using the `translate_text` function and saves the original and translated text in a result
    content string. If the action choice is not "1", it saves the transcribed text in the result content string.
    Finally, it saves the result content in a text file on the desktop using the `save_to_desktop` function.

    Note:
        - The function assumes that the `groq_client` and the `config` dictionary are defined in the global scope.
        - The function does not return anything.
    """
    base_name = os.path.basename(file_path)
    text_file_name = f"{os.path.splitext(base_name)[0]}_transcription.txt"
    
    transcribed_text = transcribe_audio(file_path, groq_client)
    if transcribed_text:
        if action_choice == "1":  # Transcribe and translate
            translated_text = translate_text(transcribed_text, content, config["openai"]["api_key"])
            result_content = f"Original: {transcribed_text}\nTranslation: {translated_text}"
        else:  # Only transcribe
            result_content = f"Transcription: {transcribed_text}"
        
        save_to_desktop(text_file_name, result_content)

def main():
    """
    The main function of the program.

    This function parses the command line arguments using the `parse_arguments` function and prints a welcome message.

    If the `content` argument is not provided or is an empty string, it prompts the user to select a language and generates the modified content using the `get_modified_content` function.

    If the `content` argument is set to 'Smart Select', it sets the content to the special content and prints the content being sent to the API.

    If the `content` argument is provided and is not 'Smart Select', it generates the modified content using the `get_modified_content` function.

    If the `file` argument is provided, it prompts the user to choose an action and a path. It checks if the action choice is None and exits the program if it is. It then processes the files in the path or the file itself using the `process_file` function.

    If the `file` argument is not provided, it creates a session folder using the `create_session_folder` function. If the `continuous` argument is provided, it enters the continuous run mode using the `continuous_run_mode` function. Otherwise, it enters the single run mode using the `single_run_mode` function.

    Parameters:
        None

    Returns:
        None
    """
    args = parse_arguments()
    print_welcome_message()

    if args.content is None or args.content == '':
        selected_language = get_language_choice(language_map)
        content = get_modified_content(selected_language)
    elif args.content == 'Smart Select':
        content = SPECIAL_CONTENT
        print(Fore.CYAN + "\nContent being sent to API (Smart Select):" + Style.RESET_ALL)
        print(content)
        print()
    else:
        content = get_modified_content(args.content)

    if args.file:
        action_choice, path = get_file_processing_choices()
        if action_choice is None:
            sys.exit(1)

        files_to_process = glob.glob(os.path.join(path, "*.wav")) if os.path.isdir(path) else [path]

        for file_path in files_to_process:
            process_file(file_path, content, action_choice)
    else:
        session_folder = create_session_folder()
        if args.continuous:
            continuous_run_mode(content, args, session_folder)
        else:
            single_run_mode(content, args, session_folder)

if __name__ == "__main__":
    main()
