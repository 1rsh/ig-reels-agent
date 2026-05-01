# Agentic Instagram Reels Classifier

An intelligent content moderation system that automatically classifies Instagram reels as SAFE, IMPLICIT_SEXUAL, or EXPLICIT_SEXUAL using vision-language models and multi-stage LLM analysis.

## Overview

This project combines:
- **Browser Automation**: Playwright-based agent to interact with Instagram reels without accumulating watch time
- **Video Captioning**: Qwen 2.5 VL (3B) vision-language model to generate detailed video descriptions
- **Multi-Stage Classification**: Three-stage LLM pipeline using GPT-4o-mini to classify sexual content with high precision

### Architecture

```
┌─────────────────────────────────────────┐
│     Instagram Reels Stream              │
└────────────────┬────────────────────────┘
                 │
        ┌────────▼────────┐
        │  ReelsAgent     │ (Browser automation)
        │                 │
        │ • Peek next     │
        │ • Download      │
        │ • Watch/Like    │
        └────────┬────────┘
                 │
        ┌────────▼────────────────────┐
        │  VideoCaptioningModel       │
        │ (Qwen 2.5-VL-3B-Instruct)   │ (Video → Text)
        └────────┬────────────────────┘
                 │
        ┌────────▼──────────────────────────┐
        │  Multi-Stage LLM Pipeline         │
        │                                   │
        │ Stage 1: Analyze (strict scan)    │
        │ Stage 2: Critique (challenge)     │
        │ Stage 3: Final Verdict (decision) │
        │                                   │
        │ Output: SAFE | IMPLICIT_SEXUAL   │
        │         | EXPLICIT_SEXUAL        │
        └────────┬──────────────────────────┘
                 │
        ┌────────▼────────┐
        │ Save Results    │
        │ (reels_data.json)
        └─────────────────┘
```

## Features

### Smart Browser Agent
- **Idle Tab Management**: Keeps Instagram backgrounded to avoid counting watch time
- **Peek & Download**: Non-invasive reel inspection via URL capture and yt-dlp
- **Session Persistence**: Reuses authenticated browser sessions across runs
- **Watch & Like**: Automatic reel engagement with 30% random like probability

### Vision-Language Video Analysis
- Extracts detailed captions from reels including:
  - Body language & posture
  - Clothing & appearance
  - Setting & props
  - Physical interactions
  - Text & audio cues
  - Explicit content analysis

### Multi-Stage Classification Engine
1. **Analysis Stage** (Temperature 0.2): Aggressive sexual cue detection
2. **Critique Stage** (Temperature 0.6): Challenges assumptions and finds gaps
3. **Final Verdict** (Temperature 0.1): Produces deterministic classification with reasoning

## Requirements

### System
- Python 3.11+
- Apple Silicon (MPS) or CUDA GPU recommended for Qwen model inference
- ~4GB VRAM for 3B model quantized operations

### Dependencies
```
playwright>=1.40.0
transformers>=4.41.0
torch>=2.0.0
openai>=1.0.0
instructor>=1.0.0
pydantic>=2.0.0
qwen-vl-utils
yt-dlp
```

## Setup

### 1. Clone and Install
```bash
git clone <repository-url>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Instagram Authentication
First run creates an interactive browser session for login:
```bash
python main.py
```
- Browser opens at `https://www.instagram.com/accounts/login/`
- Log in manually
- Session saved to `ig_session.json` for future runs

### 3. Environment Configuration
Create a `.env` file for OpenAI API:
```
OPENAI_API_KEY=sk-...
```

## Usage

### Basic Workflow
```python
import asyncio
from reels_agent import ReelsAgent, save_session, SESSION_FILE

async def main():
    # Start browser session
    async with ReelsAgent(headless=False) as agent:
        # Get current reel
        reel = await agent.current_reel()
        
        # Peek at next reel without focus change
        next_reel = await reel.seek_next()
        
        # Download video (no watch time accumulation)
        video_path = await next_reel.download("tmp/video.mp4")
        
        # Classify using multi-stage pipeline
        result = await classify_video(video_path)
        
        # Engage if content matches criteria
        if result["verdict"] in ["IMPLICIT_SEXUAL", "BORDERLINE"]:
            await next_reel.play(5)  # Watch 5 seconds
            await next_reel.like()   # 30% chance to like

asyncio.run(main())
```

### Running the Full Pipeline
```bash
python main.py
```

Processes 5 consecutive reels and saves results to `reels_data.json`:
```json
[
  {
    "url": "https://www.instagram.com/reels/...",
    "result": {
      "verdict": "IMPLICIT_SEXUAL",
      "confidence": 0.92,
      "reasoning_chain": [...],
      "key_factors_for": [...],
      "key_factors_against": [...]
    },
    "path": "tmp/uuid.mp4",
    "dataset": true
  }
]
```

## Project Structure

```
agentic/
├── main.py                 # Entry point - orchestrates reels pipeline
├── reels_agent.py          # Browser automation (Playwright)
├── classifier.py           # Vision + multi-stage LLM pipeline
├── logger.py               # Logging configuration
├── ig_session.json         # Persistent browser session (git-ignored)
├── reels_data.json         # Classification results
├── evaluate.ipynb          # Evaluation notebook
├── train.ipynb             # Training/analysis notebook
└── qwen2.5-3b-instruct-... # Model cache directory
```

## Key Components

### `ReelsAgent` (reels_agent.py)
- **Lifecycle Management**: Context manager for browser lifecycle
- **Tab Management**: Keeps Instagram in background via idle tab
- **Reel Navigation**: Scroll, download, and engage with individual reels
- **Session Persistence**: Loads/saves authentication state

### `Reel` (reels_agent.py)
- **peek** → Peek at URL without focus
- **download()** → Get video with authenticated cookies via yt-dlp
- **play(seconds)** → Bring to front, watch, return to background
- **like()** → Click like button using DOM-aware mouse positioning

### `VideoCaptioningModel` (classifier.py)
- Runs Qwen2.5-VL-3B-Instruct model locally
- Extracts rich video descriptions with 6-dimensional forensic analysis
- Optimized for MPS (Apple Silicon) or CUDA

### Classification Pipeline (classifier.py)
- **analyze()**: Extract sexual cues, euphemisms, temporal arc
- **critique()**: Challenge assumptions, find innocent readings
- **final_verdict()**: Deterministic classification with full reasoning

## Classification Labels

| Label | Definition |
|-------|-----------|
| **SAFE** | No sexual content, innuendo, or suggestive framing |
| **IMPLICIT_SEXUAL** | Sexual intent implied through euphemism, innuendo, or visual metaphor (no explicit content) |
| **EXPLICIT_SEXUAL** | Direct nudity, penetration, or unambiguous sexual acts |

## Configuration

### Model Selection
Edit `classifier.py` to use different captioning models:
```python
# Use censored model (default)
"huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"

# Or use original uncensored model
"Qwen/Qwen2.5-VL-3B-Instruct"
```

### GPU/Device Configuration
```python
# Apple Silicon
QWEN_DEVICE = "mps"

# NVIDIA GPU
QWEN_DEVICE = "cuda:0"

# CPU (slow)
QWEN_DEVICE = "cpu"
```

### LLM Model Selection
Edit `classifier.py`:
```python
MODEL = "gpt-4o-mini"  # or "gpt-4o" for higher accuracy
```
