import sounddevice as sd
import wavio
import os
import subprocess
from pynput import keyboard
from datetime import datetime
import numpy as np
import time
import sys
import logging
from colorama import Fore, Style

# Constants for recording
CHANNELS = 1
SAMPLE_WIDTH = 2
RATE = 16000
FORMAT = "int16"
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

# Global variables
audio_frames = []
is_recording = False


def record_audio(duration, session_folder):
    """
    Record audio for a specified duration.

    Args:
        duration (float): The duration of the recording in seconds.
        session_folder (str): The folder path where the recorded audio will be saved.

    Returns:
        numpy.ndarray or None: The recorded audio data as a flattened numpy array, or None if an error occurred during recording.
    """
    try:
        audio_data = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS, dtype=np.int16)
        sd.wait()
        return audio_data.flatten()
    except Exception as e:
        logging.error(f"Error during recording: {e}")
        return None

def play_audio(audio_content=None, file_path=None):
    """
    Play audio using ffplay.

    Args:
        audio_content (bytes, optional): The audio content to play. Defaults to None.
        file_path (str, optional): The path to the audio file to play. Defaults to None.

    Raises:
        Exception: If an error occurs during audio playback.
    """
    try:
        cmd = ["ffplay", "-nodisp", "-autoexit"]
        stdin_pipe = None

        if file_path:
            cmd.append(file_path)
        else:
            cmd.append("-")
            stdin_pipe = subprocess.PIPE

        ffplay_proc = subprocess.Popen(
            cmd,
            stdin=stdin_pipe,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        if audio_content and not file_path:
            ffplay_proc.stdin.write(audio_content)
            ffplay_proc.stdin.flush()
            ffplay_proc.stdin.close()

        ffplay_proc.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def save_audio_frames():
    """
    Saves the recorded audio frames to a WAV file.

    Returns:
        str: The path to the saved WAV file.
    """
    global audio_frames
    wavio.write(WAVE_OUTPUT_FILENAME, audio_frames, RATE, sampwidth=SAMPLE_WIDTH)
    return WAVE_OUTPUT_FILENAME

def voice_to_text(transcribe_func):
    """
    Transcribes spoken words from recorded audio data into text.

    This function first compiles the collected audio frames into a WAV file and then
    uses the provided transcription function to convert the audio content into text.

    Args:
        transcribe_func (callable): A function that takes a file path and returns transcribed text.

    Returns:
        str: The transcribed text as a string.

    Notes:
        - The function uses the global `audio_frames` variable for audio data.
        - It saves the audio data to a temporary WAV file before transcription.
    """
    wav_file = save_audio_frames()
    return transcribe_func(wav_file)

def clear_audio_frames():
    """Clears the global audio_frames list."""
    global audio_frames
    audio_frames = []


def record_audio_continuous():
    """
    Initiates continuous audio recording, storing incoming audio frames in memory until the recording is
    stopped. It relies on the `sounddevice.InputStream` to capture audio in real-time, with a designated
    callback function that processes and stores each audio frame.

    This function is designed to be part of a larger system that requires real-time audio data capture, such
    as a speech translation or voice command application. It will record indefinitely until the `is_recording`
    global variable is explicitly set to False, typically by another part of the application in response to
    a user command or action.

    Returns:
        bytes: A bytes object that contains all recorded audio frames concatenated together. This object can
               be written to a file, processed, or streamed further depending on the application's requirements.

    Notes:
        The recording loop runs on the main thread, with actual audio capture happening on a background thread
        managed by the `sounddevice` library. The function prints a message to the console indicating it is
        ready to record and will continue to do so until it is instructed to stop.

    Example:
        # Begin recording
        audio_data = record_audio_continuous()
        # To stop recording, set is_recording to False from another thread or signal handler.
    """
    global is_recording
    is_recording = True
    print(Fore.GREEN + "Say 'stop' to end recording..." + Style.RESET_ALL)
    with sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=record_callback):
        while is_recording:
            time.sleep(0.1)
    return b"".join(audio_frames)

def record_callback(indata, frames, time, status):
    """
    Callback function for the sounddevice.InputStream that processes incoming audio data.
    This function is called from a separate thread for each audio block captured by the
    audio input stream. It appends the captured audio frames to a global list if recording
    is active. It also handles the reporting of any audio stream statuses, such as overflows
    or underflows, which are indicators of potential issues with the recording process.

    Parameters:
        indata (numpy.ndarray): A two-dimensional NumPy array containing the captured audio data
                                for each frame, where each row represents one frame.
        frames (int): The number of frames (block size) of the audio data captured.
        time (CData): An instance of sounddevice._ffi.CData containing the timestamp of the first sample
                      in 'indata'. The structure contains 'inputBufferAdcTime' and 'currentTime' attributes.
        status (sounddevice.CallbackFlags): An instance of sounddevice.CallbackFlags indicating the status
                                            of the audio input stream.

    Notes:
        - The function modifies the global `audio_frames` list, appending new audio data if `is_recording` is True.
        - Any important `status` flags are printed to the standard error stream to alert of issues like buffer overflows.
        - This callback is designed to operate in the background, and its efficiency is crucial to avoid latency or
          loss of audio data. Therefore, operations within the callback should be kept to a minimum.
    """
    global is_recording, audio_frames
    if is_recording:
        audio_frames.append(indata.copy())
    if status:
        print(status, file=sys.stderr)

def start_recording():
    """Starts the recording process by setting is_recording to True."""
    global is_recording
    is_recording = True

def stop_recording():
    """Stops the recording process by setting is_recording to False."""
    global is_recording
    is_recording = False

# Add more audio processing functions as needed