# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import yt_dlp
import whisper


app = Flask(__name__)
CORS(app)  # allow requests from frontend

# Initialize summarizer once globally
summarizer = pipeline("summarization", model="t5-small")

# Initialize Whisper model
whisper_model = whisper.load_model("base")

def download_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "yt_audio.%(ext)s",
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.mp3"

def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def summarize_text(text):
    # T5-small has a max token limit, so limit input length if very long
    if len(text.split()) > 500:
        text = " ".join(text.split()[:500])
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Download audio from YouTube
        audio_file = download_audio(url)

        # Transcribe with Whisper
        transcript = transcribe_audio(audio_file)

        # Summarize with T5-small
        summary = summarize_text(transcript)

        return jsonify({"summary": summary})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to summarize video"}), 500

if __name__ == "__main__":
    app.run(debug=True)