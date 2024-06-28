import streamlit as st
import os
import requests
import logging
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import yaml

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load configuration and initialize Groq client
def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()
groq_api_key = config["groq"]["api_key"]

def transcribe_audio(audio_file_path):
    """
    Transcribes spoken words from an audio file into text using the Groq Whisper model.
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                },
                files={
                    "file": audio_file,
                },
                data={
                    "model": "whisper-large-v3",
                    "prompt": "Please transcribe the audio content accurately.",
                    "response_format": "json",
                    "language": "en",
                    "temperature": 0.0,
                },
            )
            response_data = response.json()
            if response.status_code == 200 and "text" in response_data:
                return response_data["text"]
            else:
                logger.error(f"Failed to transcribe audio: {response_data}\n")
                return None
    except Exception as e:
        logger.error(f"Transcription failed due to an error: {e}\n")
        return None

def get_chunk_length_ms(file_path, max_size_mb=24):
    """
    Calculates the appropriate chunk length in milliseconds based on the file size.
    """
    audio = AudioSegment.from_file(file_path)
    file_size = os.path.getsize(file_path)
    duration_ms = len(audio)
    max_size_bytes = max_size_mb * 1024 * 1024  # Slightly less than 25 MB to account for overhead
    chunk_length_ms = int(max_size_bytes / file_size * duration_ms)
    logger.info(f"Calculated chunk length: {chunk_length_ms} ms")
    return chunk_length_ms

def split_audio(file_path, chunk_length_ms):
    """
    Splits the audio file into smaller chunks.
    """
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_file_path = f"{file_path.stem}_chunk{i // chunk_length_ms}.wav"
        chunk.export(chunk_file_path, format="wav")
        chunks.append(chunk_file_path)
    return chunks

def main():
    """
    Main function to handle audio transcription.
    """
    st.title("Audio Transcription App")
    st.write("Enter the path to an audio file to transcribe:")

    file_path = st.text_input("File path")

    if st.button("Transcribe"):
        if not file_path:
            st.error("Please provide the path to an audio file.")
            return

        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file() or file_path.suffix != ".wav":
            st.error("Invalid file path. Please provide a valid .wav file.")
            return

        base_name = file_path.stem
        text_file_name = f"{base_name}_transcription.txt"

        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) // 1000
        st.write(f"The audio file is {duration_seconds} seconds long.")
        if duration_seconds > 600:  # 10 minutes
            user_input = st.radio("The audio is longer than 10 minutes. Do you want to split it into smaller chunks for transcription?", ("Yes", "No"))
            if user_input != "Yes":
                st.error("Transcription aborted by user.")
                return

        chunk_length_ms = get_chunk_length_ms(file_path)
        chunks = split_audio(file_path, chunk_length_ms)

        all_transcriptions = []
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            transcribed_text = transcribe_audio(chunk)
            if transcribed_text:
                all_transcriptions.append(transcribed_text)
            os.remove(chunk)  # Clean up chunk file after transcription
            progress_bar.progress((i + 1) / len(chunks))

        full_transcription = "\n".join(all_transcriptions)
        st.text_area("Transcription", full_transcription, height=300)

if __name__ == "__main__":
    main()
