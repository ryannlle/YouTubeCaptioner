from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import whisper
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

client = OpenAI()
WHISPER_MODEL = whisper.load_model("base")  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOMEPAGE_PATH = os.path.join(BASE_DIR, "..", "frontend", "homepage.html")
PLAYER_PATH = os.path.join(BASE_DIR, "..", "frontend", "player.html")

def get_youtube_info_and_subs(url):
    """Fetch video metadata and subtitles info safely (no download)."""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info

def get_youtube_transcript(info):
    """Try to fetch YouTube captions if available."""
    subtitles = info.get("requested_subtitles") or info.get("automatic_captions")
    if not subtitles:
        return None
    sub_info = subtitles.get("en") or next(iter(subtitles.values()), None)
    if not sub_info or "url" not in sub_info:
        return None
    r = requests.get(sub_info["url"])
    return r.text

def vtt_to_text(vtt_content):
    """Convert VTT caption content to plain text."""
    lines = vtt_content.splitlines()
    text_lines = [line for line in lines if line and not line.startswith("WEBVTT") and "-->" not in line]
    return " ".join(text_lines)

def download_audio(url):
    """Download and extract audio as MP3."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "yt_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.mp3"

def transcribe_audio(audio_file):
    """Transcribe audio using Whisper (auto language detection)."""
    result = WHISPER_MODEL.transcribe(audio_file)
    return result["text"]

def process_with_gpt(text, prompt=None, language="English"):
    """Generate a structured Markdown study guide using GPT in the requested language."""
    base_prompt = (
        f"You are an expert educator. Please write the following STUDY GUIDE in **{language}**.\n\n"
        "Use Markdown with headings and bullet points only — no long paragraphs.\n\n"
        "Follow this exact structure:\n\n"
        "## Overview\n"
        "- One-sentence summary\n\n"
        "## Key Concepts\n"
        "- Core ideas as bullet points\n\n"
        "## Detailed Notes\n"
        "- Step-by-step breakdown\n\n"
        "## Examples\n"
        "- Real-world examples (if any)\n\n"
        "## Summary / Takeaways\n"
        "- 4–6 short bullets highlighting key lessons\n\n"
        f"Transcript:\n{text}"
    )

    if prompt:
        base_prompt += f"\n\nExtra instruction: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

@app.route("/")
def home():
    return send_file(HOMEPAGE_PATH)

@app.route("/player")
def player():
    return send_file(PLAYER_PATH)

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    custom_prompt = data.get("prompt")
    language = data.get("language", "English") 
    if not url:
        return jsonify({"error": "No YouTube URL provided."}), 400

    try:
        info = get_youtube_info_and_subs(url)
        title = info.get("title", "Unknown Video")

    
        transcript = get_youtube_transcript(info)
        if transcript:
            text = vtt_to_text(transcript)
        else:
           
            audio_file = download_audio(url)
            text = transcribe_audio(audio_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)

        
        summary = process_with_gpt(text, prompt=custom_prompt, language=language)

        return jsonify({"summary": summary, "title": title})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to summarize the video."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
