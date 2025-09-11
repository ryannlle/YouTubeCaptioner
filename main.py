import yt_dlp
import whisper

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
print(result["text"])