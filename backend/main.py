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
    lines = vtt_content.splitlines()
    text_lines = [line for line in lines if line and not line.startswith("WEBVTT") and "-->" not in line]
    return " ".join(text_lines)

def download_audio(url):
    """Fallback if no captions."""
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
    result = WHISPER_MODEL.transcribe(audio_file)
    return result["text"]

def process_with_gpt(text, prompt=None):
    base_prompt = (
        f"You are an expert at making educational study guides. "
        f"Summarize the following transcript in Markdown with headings, bullet points, and examples:\n\n{text}"
    )
    if prompt:
        base_prompt += f"\n\nAdditional instruction: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=800,
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

        summary = process_with_gpt(text, prompt=custom_prompt)
        return jsonify({"summary": summary, "title": title})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to summarize the video."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
