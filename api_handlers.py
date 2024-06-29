from groq import Groq
from openai import OpenAI
import requests
import os
import logging
import time
from colorama import Fore, Style
from datetime import datetime

logger = logging.getLogger(__name__)

def transcribe_audio(audio_file_path, client):
    """Transcribe audio using Groq API."""
    try:
        logging.info(f"Transcribing audio file: {audio_file_path}")
        with open(audio_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), audio_file),
                model="whisper-large-v3",
                prompt="Please focus solely on transcribing the content of this audio. Do not translate. Maintain the original language and context as accurately as possible.",
                response_format="json",
                language="en",
                temperature=0.4
            )
            logging.info(f"Transcription response: {response}")
            return response.text
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return None

def translate_text(text, content, openai_api_key):
    """Translate text using OpenAI API."""
    try:
        logging.info(f"Translating text: {text}")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            },
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": content},
                    {"role": "user", "content": f"{text}"},
                ],
            },
        )

        if response.status_code == 200:
            response_data = response.json()
            translated_text = response_data["choices"][0]["message"]["content"].strip()
            logging.info(f"Translation response: {translated_text}")
            return translated_text
        else:
            logging.error(f"Failed to translate text: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        return None
    

def voice_stream(input_text, chosen_voice, session_folder, client, play_audio_func):
    """
    Converts the given text into speech using the specified voice, through the OpenAI API's text-to-speech synthesis.
    The generated speech is both played immediately and saved as an audio file in the specified session folder.

    This function is integral to applications requiring audio feedback or interaction, particularly in scenarios
    like language translation where the translated text is vocalized. It serves the dual purpose of providing
    immediate audio playback and storing the audio for possible replay.

    Args:
        input_text (str): The text that needs to be converted into speech.
        chosen_voice (str): The identifier of the voice model to be used for the synthesis.
        session_folder (str): The directory path where the synthesized audio file will be saved.
        client (OpenAI): The OpenAI client instance.
        play_audio_func (function): A function to play the audio content.

    Returns:
        str: The file path of the saved AI audio file, allowing for subsequent access and replay.

    Raises:
        Exception: An exception is raised and logged if there's an error during the synthesis process.

    Notes:
        The function first synthesizes the speech and then saves the audio content in a WAV file within the session folder.
        The filename includes a timestamp to ensure uniqueness. After saving, the function also plays the audio file for immediate feedback.
    """
    try:
        response = client.audio.speech.create(
            model="tts-1", voice=chosen_voice, input=input_text
        )
        ai_audio_filename = f"ai_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        ai_audio_path = os.path.join(session_folder, ai_audio_filename)
        with open(ai_audio_path, "wb") as f:
            f.write(response.content)
        print(f"AI voice saved in {ai_audio_path}")
        play_audio_func(response.content)  # Play the audio
    except Exception as e:
        logger.error(Fore.RED + f"Failed to speak text: {e}\n")
        return None
    return ai_audio_path  # Return the path to the saved AI audio file

# Add more API-related functions as needed