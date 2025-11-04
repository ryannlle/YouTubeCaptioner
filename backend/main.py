from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import whisper
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI()

# Load Whisper model globally
WHISPER_MODEL = whisper.load_model("base")

# ---------------- Path Setup ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOMEPAGE_PATH = os.path.join(BASE_DIR, "..", "frontend", "homepage.html")
PLAYER_PATH = os.path.join(BASE_DIR, "..", "frontend", "player.html")

# ---------------- Helper Functions ----------------

def get_youtube_transcript(url):
    """Try to fetch YouTube captions first. Return None if not available."""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get("requested_subtitles") or info.get("automatic_captions")
        if not subtitles:
            return None
        # Pick English if available
        sub_info = subtitles.get("en") or next(iter(subtitles.values()))
        if not sub_info or "url" not in sub_info:
            return None
        r = requests.get(sub_info["url"])
        return r.text

def vtt_to_text(vtt_content):
    """Convert VTT subtitle text to plain text."""
    lines = vtt_content.splitlines()
    text_lines = [line for line in lines if line and not line.startswith("WEBVTT") and "-->" not in line]
    return " ".join(text_lines)

def download_audio(url):
    """Fallback: download audio if captions are not available."""
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
    """Transcribe audio with Whisper."""
    result = WHISPER_MODEL.transcribe(audio_file)
    return result["text"]

def process_with_gpt(text, task="summarize", prompt=None):
    """Processes the transcript using GPT (Markdown-formatted output)."""
    if task != "summarize":
        raise ValueError("Only 'summarize' task is supported.")

    base_prompt = (
<<<<<<< Updated upstream
        "You are an expert educator. Turn the following lecture transcript into a clear STUDY GUIDE.\n"
        "The output **must** use Markdown with headings and bullet points — no paragraphs of plain text.\n\n"
        "Follow this exact format:\n\n"
        "## Overview\n"
        "- One-sentence summary of the topic\n\n"
        "## Key Concepts\n"
        "- Main ideas explained simply\n"
        "- Each key idea on its own bullet\n\n"
        "## Detailed Notes\n"
        "- Step-by-step bullet notes\n"
        " - Use sub-bullets for examples or details\n\n"
        "## Examples\n"
        "- Real-world or practical examples (if applicable)\n\n"
        "## Summary / Takeaways\n"
        "- 4–6 short bullet points highlighting the key lessons\n\n"
        "Make sure every list actually uses dash bullets ('-') and no numbered lists.\n\n"
        f"Transcript:\n{text}"
=======
        f"You are an expert at making educational study guides. "
        f"Summarize the following transcript in a clean, readable format using Markdown with headings, bullet points, and examples:\n\n{text}"
>>>>>>> Stashed changes
    )

    if prompt:
        base_prompt += f"\n\nExtra instruction: {prompt}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Make sure you have access to this model
        messages=[{"role": "user", "content": base_prompt}],
        max_tokens=1000,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

# ---------------- Flask Routes ----------------
@app.route("/")
def home():
    if not os.path.exists(HOMEPAGE_PATH):
        return "Homepage not found!", 404
    return send_file(HOMEPAGE_PATH)

@app.route("/player")
def player():
    if not os.path.exists(PLAYER_PATH):
        return "Player page not found!", 404
    return send_file(PLAYER_PATH)

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    custom_prompt = data.get("prompt")

    if not url:
        return jsonify({"error": "No YouTube URL provided."}), 400

    try:
        # Step 1: try captions
        transcript = get_youtube_transcript(url)
        if transcript:
            text = vtt_to_text(transcript)
        else:
            # Step 2: fallback to Whisper transcription
            audio_file = download_audio(url)
            text = transcribe_audio(audio_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)

        # Step 3: summarize
        summary = process_with_gpt(text, task="summarize", prompt=custom_prompt)
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to summarize the video."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
