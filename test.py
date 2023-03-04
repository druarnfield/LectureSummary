import os
import openai
import filechunkio
import nltk
from pydub import AudioSegment
from moviepy.editor import VideoFileClip


# Path to the directory where the video files are stored
lectures_dir = "Lectures/"

# Get a list of all the files in the directory
folder_files = os.listdir(lectures_dir)

# Filter the list to only include MP4 files
lecture_files = [f"{lectures_dir}{file}" for file in folder_files if file.endswith(".mp4")]

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def convert_videos_to_mp3(videos):
    """
    Converts a list of video files to MP3 files, and splits the MP3 files into 20-minute chunks if they're larger than 25 MB.

    Args:
        videos (list): A list of strings, each string representing the path to a video file.

    Returns:
        A dictionary where the keys are the video file paths and the values are lists of strings, each string representing
        the path to an MP3 file.
    """
    results = {}
    for video_file_path in videos:
        # Load the video file
        clip = VideoFileClip(video_file_path)

        # Extract the audio from the video
        audio = clip.audio

        # Save the audio to an MP3 file
        audio_file_path = os.path.join("Audio", f"{os.path.basename(video_file_path)}.mp3")
        audio.write_audiofile(audio_file_path)

        # Create a list with the name of the MP3 file
        filenames = [audio_file_path]

        # Set the maximum file size in bytes (25 MB)
        max_size = 25 * 1024 * 1024 

        # Check if the MP3 file is larger than the maximum size
        if os.path.getsize(filenames[0]) > max_size:
            # Load the MP3 file
            audio = AudioSegment.from_file(filenames[0], format="mp3")
            # Split the audio into chunks
            filenames = split_audio_into_chunks(audio, video_file_path)
            # Remove the original file
            os.remove(audio_file_path)

        # Add the results to the dictionary
        results[video_file_path] = filenames

    return results

def split_audio_into_chunks(audio, video_file_path):
    """
    Splits an AudioSegment object into 20-minute chunks and exports each chunk to a separate MP3 file.

    Args:
        audio (AudioSegment): The audio to split.
        video_file_path (str): The path to the video file.

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
        folder_path = os.path.join("Audio", os.path.basename(video_file_path).split(".")[0])
        os.makedirs(folder_path, exist_ok=True)
        chunk_file_path = os.path.join(folder_path, f"chunk_{i}.mp3")
        chunk.export(chunk_file_path, format="mp3")
        filenames.append(chunk_file_path)

    return filenames


def convert_mp3_to_text(mp3_files):
    """
    Transcribes each MP3 file in a dictionary to text using OpenAI API, and saves the text to a file.

    Args:
        mp3_files (dict): A dictionary where the keys are the video file paths and the values are lists of strings,
                          each string representing the path to an MP3 file.

    Returns:
        A list of the text file paths created.
    """
    text_files = []
    for video_path, mp3_path_list in mp3_files.items():
        for mp3_file_path in mp3_path_list:
            # Open the file in binary mode
            audio_file = open(mp3_file_path, "rb")
            # Transcribe the audio using OpenAI API
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            # Write the transcript to a text file
            output_file_path = os.path.join("Output", f"{os.path.basename(video_path)}.txt")
            with open(output_file_path, "a") as f:
                f.write(transcript['text'])
            text_files.append(output_file_path)
    return text_files
                

def split_text_into_chunks(text_file_path):
    """
    Splits a text file into 4000 character chunks and saves each chunk to a separate file in a folder named after the
    original file (without the extension).

    Args:
        text_file_path (str): The path to the text file.

    Returns:
        A list of strings, each string representing the path to a chunk file.
    """
    # Create a folder for the chunk files
    folder_path = os.path.splitext(text_file_path)[0]
    os.makedirs(folder_path, exist_ok=True)

    # Set the chunk size to 4000 characters
    chunk_size = 4000

    # Open the text file in binary mode
    with open(text_file_path, "rb") as f:
        # Initialize the start byte and chunk index
        start_byte = 0
        chunk_index = 0
        while True:
            # Read a chunk of data from the file
            chunk = filechunkio.FileChunkIO(f, "r", start=start_byte, chunksize=chunk_size)
            data = chunk.read()

            # If there's no more data, break out of the loop
            if not data:
                break

            # Write the chunk to a new file in the folder
            chunk_file_path = os.path.join(folder_path, f"chunk_{chunk_index}.txt")
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(data)

            # Update the start byte and chunk index
            start_byte += len(data)
            chunk_index += 1

    # Get a list of the chunk file paths
    chunk_file_paths = [os.path.join(folder_path, f"chunk_{i}.txt") for i in range(chunk_index)]

    return chunk_file_paths


def split_text_files(files, max_tokens=2000):
    """
    Splits a list of text files into chunks of approximately `max_tokens` tokens and saves them to
    individual files in a directory with the same name as the original file (minus the extension).

    Args:
        files (list): A list of strings representing the file paths of the text files to split.
        max_tokens (int): The maximum number of tokens per chunk (default: 4000).

    Returns:
        A dictionary where the keys are the folder names where the output files were saved, and the values
        are lists of strings representing the file paths of the output chunks.
    """
    output = {}
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for file in files:
        # Create a directory with the same name as the input file (minus the extension)
        folder_name = os.path.splitext(os.path.basename(file))[0]
        os.makedirs(os.path.join("SplitText", folder_name), exist_ok=True)
        # Read in the input file
        with open(file, 'r') as f:
            text = f.read()
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        # Split the tokens into chunks of approximately `max_tokens` tokens
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        # Save each chunk to a separate file
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_text = ' '.join(chunk)
            chunk_path = os.path.join("SplitText", folder_name, f'{i}.txt')
            with open(chunk_path, 'w') as f:
                f.write(chunk_text)
            chunk_paths.append(chunk_path)
        # Add folder name and chunk paths to output dictionary
        output[folder_name] = chunk_paths
    return output

def summarize_text(folder_files_dict):
    """
    Summarizes the text in a file using OpenAI's CHAT-GPT API.

    Args:
        file_path (str): The file path of the text file to summarize.

    Returns:
        A string representing the summary of the text.
    """

    for folder_name, file_list in folder_files_dict.items():
        for file_path in file_list:
            # Read in the input file
            with open(file_path, 'r') as f:
                text = f.read()
             # Set up the message history for the chat completion API
             # Use the Chat-GPT API to summarize the text
            response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
            {"role": "system", "content": "You are a helpful assistant who summarises text for a professor."},
            {"role": "user", "content": "Summarise this lecture in markdown.  Use latex for math symbols.  this is not the full lecture : " + text},
            ]
            )
            # Extract the summary from the response
            summary = response['choices'][0]['message']['content']
            
            # Save the summary to a file in the same folder as the input file
            summary_file_path = os.path.join(os.path.dirname(folder_name), 'summary.txt')
            with open(summary_file_path, 'a') as f:
                f.write(summary)


if __name__ == "__main__":
    # Convert the video file to an MP3 file
    mp3_file_path = convert_videos_to_mp3(lecture_files)
    # Transcribe the MP3 file to text
    text_files = convert_mp3_to_text(mp3_file_path)
    text_files = split_text_files(text_files)
    summarize_text(text_files)

