import os
import openai
import json
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def convert_video_to_mp3(video_file_path):
    """
    Converts a video file to an MP3 file, and splits the MP3 file into 20-minute chunks if it's larger than 25 MB.

    Args:
        video_file_path (str): The path to the video file.

    Returns:
        A list of strings, each string representing the path to an MP3 file.
    """
    # Load the video file
    clip = VideoFileClip(video_file_path)

    # Extract the audio from the video
    audio = clip.audio

    # Save the audio to an MP3 file
    audio_file_path = os.path.join("Audio", "my_audio.mp3")
    audio.write_audiofile(audio_file_path)

    # Create a list with the name of the MP3 file
    filename = [audio_file_path]

    # Set the maximum file size in bytes (25 MB)
    max_size = 25 * 1024 * 1024 

    # Check if the MP3 file is larger than the maximum size
    if os.path.getsize(filename[0]) > max_size:
        # Load the MP3 file
        audio = AudioSegment.from_file(filename[0], format="mp3")
        # Split the audio into chunks
        filename = split_audio_into_chunks(audio)
        # Remove the original file
        os.remove(filename[0])

    return filename

def split_audio_into_chunks(audio):
    """
    Splits an AudioSegment object into 20-minute chunks and exports each chunk to a separate MP3 file.

    Args:
        audio (AudioSegment): The audio to split.

    Returns:
        A list of strings, each string representing the path to an MP3 file.
    """
    duration = audio.duration_seconds
    chunk_size = 20 * 60  # 20 minutes in seconds
    num_chunks = int(duration / chunk_size) + 1
    # Split the audio into chunks
    filenames = []
    for i in range(num_chunks):
        start = i * chunk_size * 1000
        end = (i + 1) * chunk_size * 1000
        chunk = audio[start:end]
        # Export the chunk to a separate MP3 file
        chunk_file_path = os.path.join("Audio", f"chunk_{i}.mp3")
        chunk.export(chunk_file_path, format="mp3")
        filenames.append(chunk_file_path)

    return filenames

def convert_mp3_to_text(mp3_file_path):
    """
    Transcribes each MP3 file to text using OpenAI API, and saves the text to a file.

    Args:
        mp3_file_path (list): A list of strings, each string representing the path to an MP3 file.
    """
    # Transcribe each file in the filename list
    for file in mp3_file_path:
        # Open the file in binary mode
        audio_file = open(file, "rb")
        # Transcribe the audio using OpenAI API
        transcript = json.dumps(openai.Audio.transcribe("whisper-1", audio_file))
        # Write the transcript to a text file
        output_file_path = os.path.join("Output", "transcript.txt")
        with open(output_file_path, "a") as f:
            f.write(transcript)
