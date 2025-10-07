from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import whisper
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Initialize Hugging Face pipelines globally
summarizer = pipeline("summarization", model="pszemraj/led-base-book-summary")
caption_generator = pipeline("text2text-generation", model="google/flan-t5-base")

WHISPER_MODEL = whisper.load_model("base")  # load once globally

# ---------------- Audio & Transcription ----------------
def download_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "yt_audio.%(ext)s",
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
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

def process_with_llm(text, task="summarize", prompt=None):
    if task == "summarize":
        base_summary = summarizer(
            text,
            max_new_tokens=512,
            min_length=120,
            no_repeat_ngram_size=3,
            do_sample=False,
        )[0]["summary_text"]

        if prompt:
            generation_input = f"Instruction: {prompt}\n\nContext: {base_summary}\n\nRespond with a detailed answer."
            refined = caption_generator(
                generation_input,
                max_new_tokens=256,
                do_sample=False,
            )[0]["generated_text"]
            return refined

        return base_summary
    elif task == "caption":
        caption = caption_generator(f"Generate a caption: {text}", max_length=50)
        return caption[0]['generated_text']
    else:
        raise ValueError("Invalid task. Use 'summarize' or 'caption'.")

# ---------------- Flask API ----------------
@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    custom_prompt = data.get("prompt")  # optional

    if not url:
        return jsonify({"error": "No YouTube URL provided."}), 400

    try:
        audio_file = download_audio(url)
        text = transcribe_audio(audio_file)
        summary = process_with_llm(text, task="summarize", prompt=custom_prompt)
        
        # Cleanup audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to summarize the video."}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)