import yt_dlp
import whisper
from transformers import pipeline

# Initialize Hugging Face pipelines globally
summarizer = pipeline("summarization", model="pszemraj/led-base-book-summary")
caption_generator = pipeline("text2text-generation", model="google/flan-t5-base")

url = "https://www.youtube.com/watch?v=qg4PchTECck" # Example YouTube URL

def download_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "yt_audio.%(ext)s",
        "quiet": False,
        # Request Android player manifests to avoid SABR streams with missing audio
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    print("\nDownloading audio...\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.download([url])

def transcribe_audio(audio):
    print("\nTranscribing audio...\n")
    model = whisper.load_model("base")
    result = model.transcribe("yt_audio.mp3")
    transcribed_text = result["text"]
    return transcribed_text

def process_with_llm(text, task="summarize", prompt=None):
    print("\nSummarizing...\n")
    if task == "summarize":
        base_summary = summarizer(
            text,
            max_new_tokens=512,
            min_length=120,
            no_repeat_ngram_size=3,
            do_sample=False,
        )[0]["summary_text"]

        if prompt:
            generation_input = (
                f"Instruction: {prompt}\n\nContext: {base_summary}\n\nRespond with a detailed answer."
            )
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



url = input("Enter YouTube URL: ")

audio = download_audio(url)
transcribed_text = transcribe_audio(audio)
custom_prompt = input("Enter a summarization prompt (or leave blank for default): ")
summary = process_with_llm(transcribed_text, task="summarize", prompt=custom_prompt if custom_prompt else None)

print("Summary:", summary)
