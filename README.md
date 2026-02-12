# YouTubeCaptioner

A full-stack web application that transforms YouTube videos into structured, AI-generated study guides. Paste any YouTube URL and get a clean, Markdown-formatted summary — with multilingual support and PDF export.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-Speech--to--Text-74aa9c)

## Features

- **YouTube transcript extraction** — pulls existing captions via `yt-dlp` when available, avoiding unnecessary compute
- **Whisper fallback** — automatically downloads audio and transcribes with OpenAI Whisper when no captions exist
- **AI-powered summarization** — generates structured study guides (Overview, Key Concepts, Detailed Notes, Examples, Takeaways) using GPT-4o-mini
- **Multilingual output** — summaries in English, Spanish, French, or Chinese
- **Custom prompts** — users can steer the summarization with optional instructions
- **Summary history** — browser-side persistence via `localStorage` with one-click recall
- **PDF export** — download any summary as a formatted PDF
- **Embedded video player** — watch the original video alongside its summary

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                          │
│  homepage.html ──── Input form, history, animations  │
│  player.html   ──── Video embed, summary view, PDF   │
└────────────────────────┬────────────────────────────┘
                         │  POST /api/summarize
                         ▼
┌─────────────────────────────────────────────────────┐
│                  Backend (Flask)                      │
│                                                      │
│  1. Extract video info ───── yt-dlp                  │
│  2. Get transcript:                                  │
│     ├─ Try YouTube captions ─ yt-dlp + requests      │
│     └─ Fallback: audio ───── yt-dlp → Whisper        │
│  3. Summarize ────────────── OpenAI GPT-4o-mini      │
│  4. Return JSON ──────────── { summary, title }      │
└─────────────────────────────────────────────────────┘
```

### Project Structure

```
YouTubeCaptioner/
├── backend/
│   └── main.py              # Flask server, API routes, AI pipeline
├── frontend/
│   ├── homepage.html         # Landing page with URL input and history
│   └── player.html           # Video player and rendered summary
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) (required by Whisper for audio processing)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/JZaelit/YouTubeCaptioner.git
   cd YouTubeCaptioner
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the server**

   ```bash
   python backend/main.py
   ```

5. **Open the app** at [http://localhost:5000](http://localhost:5000)

## API Reference

### `POST /api/summarize`

Accepts a YouTube URL and returns an AI-generated study guide.

**Request body:**

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "prompt": "Focus on the musical composition techniques",
  "language": "English"
}
```

| Field      | Type   | Required | Description                                      |
|------------|--------|----------|--------------------------------------------------|
| `url`      | string | Yes      | Any valid YouTube URL                             |
| `prompt`   | string | No       | Custom instruction to guide the summarization     |
| `language` | string | No       | Output language: `English`, `Spanish`, `French`, `Chinese` (default: `English`) |

**Response:**

```json
{
  "summary": "## Overview\n- ...",
  "title": "Video Title"
}
```

## Tech Stack

| Layer    | Technology                | Purpose                              |
|----------|---------------------------|--------------------------------------|
| Backend  | Flask                     | HTTP server and API routing          |
| Backend  | yt-dlp                    | YouTube metadata and caption extraction |
| Backend  | OpenAI Whisper (base)     | Speech-to-text fallback              |
| Backend  | OpenAI GPT-4o-mini        | Transcript summarization             |
| Frontend | Vanilla HTML/CSS/JS       | UI — no framework overhead           |
| Frontend | marked.js                 | Markdown rendering                   |
| Frontend | html2canvas + jsPDF       | Client-side PDF generation           |
| Storage  | Browser localStorage      | Summary history persistence          |

## How the Summarization Pipeline Works

1. **Metadata extraction** — `yt-dlp` fetches the video title and available subtitle tracks without downloading the video itself.

2. **Transcript retrieval** — The app first checks for existing YouTube captions (manual or auto-generated). If captions are found, they are downloaded as VTT and converted to plain text. This path is fast and avoids heavy processing.

3. **Whisper fallback** — If no captions are available, the app downloads the best available audio stream, converts it to MP3 via FFmpeg, and runs it through OpenAI's Whisper `base` model for transcription. The temporary audio file is cleaned up afterward.

4. **GPT summarization** — The transcript is sent to GPT-4o-mini with a structured prompt that produces a study guide in Markdown format. The prompt enforces a consistent structure (Overview, Key Concepts, Detailed Notes, Examples, Summary/Takeaways) and respects the user's language choice and any custom instructions.

## License

This project is open source. See the repository for license details.
