
import os
import openai
import json
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load the video file
clip = VideoFileClip("my_video.mp4")

# Extract the audio from the video
audio = clip.audio

# Save the audio to an MP3 file
audio.write_audiofile("Audio/my_audio.mp3")

# Create a list with the name of the MP3 file
filename = ['Audio/my_audio.mp3']

# Set the maximum file size in bytes (25 MB)
max_size = 25 * 1024 * 1024 

# Check if the MP3 file is larger than the maximum size
if os.path.getsize(filename[0]) > max_size:
    # Load the MP3 file
    audio = AudioSegment.from_file(filename[0], format="mp3")
    duration = audio.duration_seconds
    chunk_size = 20 * 60  # 20 minutes in seconds
    num_chunks = int(duration / chunk_size) + 1
    # Split the audio into chunks
    for i in range(num_chunks):
        start = i * chunk_size * 1000
        end = (i + 1) * chunk_size * 1000
        chunk = audio[start:end]
        # Export the chunk to a separate MP3 file
        chunk.export(f"Audio/chunk_{i}.mp3", format="mp3")
    # Remove the original file
    os.remove(filename[0])
    # Update the filename list with the names of the new files
    filename = [f"Audio/chunk_{i}.mp3" for i in range(num_chunks)]
else:
    # If the file is smaller than the maximum size, do nothing
    pass

# Transcribe each file in the filename list
for file in filename:
    # Open the file in binary mode
    audio_file = open(file, "rb")
    # Transcribe the audio using OpenAI API
    transcript = json.dumps(openai.Audio.transcribe("whisper-1", audio_file))
    # Write the transcript to a text file
    output_file = open("Output/transcript.txt", "a")   
    with output_file as f:
        f.write(transcript)



