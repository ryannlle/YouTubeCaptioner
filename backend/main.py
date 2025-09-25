import yt_dlp
import whisper
from transformers import pipeline
from peft import PeftModel

# Download audio from YouTube video

url = "https://www.youtube.com/watch?v=qg4PchTECck"

ydl_opts = {"quiet": True}

ydl_opts = {
    "format": "bestaudio/best",     # best audio quality 
    "outtmpl": "yt_audio.%(ext)s",    #save as yt_audion.mp3
    "quiet":False,
    "postprocessors": [
        {
            # converts to mp3 with ffmpeg
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

# Transcribe audio using Whisper

model = whisper.load_model("base")
result = model.transcribe("yt_audio.mp3")
#print(result["text"])

# Feed into Hugging Face Transformers  LLM

# Initialize Hugging Face pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
caption_generator = pipeline("text2text-generation", model="t5-small")

transcribed_text = result["text"]

def process_with_llm(text, task="summarize"):
    if task == "summarize":
        # Use the summarization pipeline
        summary = summarizer(text, max_length=500, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    elif task == "caption":
        # Use the caption generation pipeline
        caption = caption_generator(f"Generate a caption: {text}", max_length=50)
        return caption[0]['generated_text']
    else:
        raise ValueError("Invalid task. Use 'summarize' or 'caption'.")

summary = process_with_llm(transcribed_text, task="summarize")

print("Summary:", summary)