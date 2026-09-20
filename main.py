"""
Audio analysis server.

Single endpoint:
  POST /api/analyze  -> upload an audio file, get back detected key + chord progression

Usage:
    uvicorn main:app --reload

Test:
    curl -X POST http://localhost:8000/api/analyze -F "audio=@song.wav"
"""

import logging
import os
import tempfile

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# ---------------------------------------------------------------------------
# Chord recognition
# ---------------------------------------------------------------------------

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def make_chord_templates():
    """Build 24 binary triad templates (12 major + 12 minor)."""
    templates = {}
    major_intervals = [0, 4, 7]
    minor_intervals = [0, 3, 7]
    for i, root in enumerate(NOTES):
        maj = np.zeros(12)
        minr = np.zeros(12)
        for interval in major_intervals:
            maj[(i + interval) % 12] = 1
        for interval in minor_intervals:
            minr[(i + interval) % 12] = 1
        templates[f'{root}'] = maj
        templates[f'{root}m'] = minr
    return templates


def recognize_chords(chroma, templates):
    """Match each chroma column against chord templates via cosine similarity."""
    chord_names = list(templates.keys())
    template_matrix = np.array([templates[c] for c in chord_names])

    # Normalize so only the *shape* of the pitch-class distribution matters
    template_matrix = template_matrix / (np.linalg.norm(template_matrix, axis=1, keepdims=True) + 1e-8)
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)

    similarities = template_matrix @ chroma_norm  # (24, n_frames)
    best_chords = np.argmax(similarities, axis=0)

    return [chord_names[i] for i in best_chords]


# ---------------------------------------------------------------------------
# Key detection (Krumhansl-Schmuckler algorithm)
# ---------------------------------------------------------------------------

# Krumhansl-Kessler key profiles: empirically-derived weights for how strongly
# each scale degree is expected to appear in a given key (tonic/dominant
# weighted highest, not just binary scale membership).
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def make_key_templates():
    """Build 24 rotated key profile templates (12 major + 12 minor)."""
    templates = {}
    for i, root in enumerate(NOTES):
        templates[f'{root} major'] = np.roll(MAJOR_PROFILE, i)
        templates[f'{root} minor'] = np.roll(MINOR_PROFILE, i)
    return templates


def detect_key(chroma_mean, templates):
    """Match an aggregated chroma vector against key templates via cosine similarity."""
    key_names = list(templates.keys())
    template_matrix = np.array([templates[k] for k in key_names])

    template_matrix = template_matrix / np.linalg.norm(template_matrix, axis=1, keepdims=True)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)

    similarities = template_matrix @ chroma_norm  # (24,)
    best_idx = np.argmax(similarities)

    return key_names[best_idx], similarities[best_idx], dict(zip(key_names, similarities))


# ---------------------------------------------------------------------------
# Core analysis (pure function: audio in, structured results out)
# ---------------------------------------------------------------------------

def analyze_audio(y: np.ndarray, sr: int) -> dict:
    """
    Run full analysis on a loaded waveform:
      1. Extract chroma features
      2. Beat-sync chroma for stable, musically-meaningful chord segments
      3. Recognize chord per beat
      4. Detect overall key from whole-track chroma

    Returns a dict of plain Python types (no numpy), ready for JSON.
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Chroma extraction
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    # Beat-synchronous chroma for stable chord estimates
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

    # librosa.util.sync can return one extra leading column (segment before
    # the first beat) — pad beat_times with 0.0 to keep them aligned.
    if chroma_sync.shape[1] == len(beat_times) + 1:
        beat_times = np.concatenate([[0.0], beat_times])

    # Chord recognition (beat-level), collapsed to only show chord changes
    chord_templates = make_chord_templates()
    chords_per_beat = recognize_chords(chroma_sync, chord_templates)

    chord_progression = []
    prev_chord = None
    for t, chord in zip(beat_times, chords_per_beat):
        if chord != prev_chord:
            chord_progression.append({"time": float(t), "chord": chord})
            prev_chord = chord

    # Key detection (whole-track)
    chroma_mean = np.mean(chroma, axis=1)
    key_templates = make_key_templates()
    key, score, all_scores = detect_key(chroma_mean, key_templates)

    # In some librosa versions, beat_track returns tempo as a 1-element
    # (or occasionally multi-element) array rather than a plain scalar.
    # np.asarray(...).flatten()[0] handles scalar, 0-d, and 1-d array cases.
    tempo_value = float(np.asarray(tempo).flatten()[0])

    return {
        "duration_seconds": round(float(duration), 2),
        "tempo_bpm": round(tempo_value, 1),
        "key": key,
        "key_confidence": round(float(score), 4),
        "chords": chord_progression,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    """
    Accepts an uploaded audio file (multipart/form-data, field name "audio")
    and returns its detected key and chord progression as JSON.
    """
    contents = await audio.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Write to a temp file (preserving the original extension) rather than
    # loading from an in-memory buffer. librosa's fallback decoder
    # (audioread -> ffmpeg) shells out to the ffmpeg binary, which needs an
    # actual file path to read from reliably for containers like webm/mp4 -
    # in-memory BytesIO can silently fail that handoff.
    suffix = os.path.splitext(audio.filename or "")[1] or ".tmp"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        try:
            y, sr = librosa.load(tmp_path, sr=22050)
        except Exception as exc:
            logger.exception("Failed to decode uploaded audio")
            raise HTTPException(status_code=422, detail=f"Could not decode audio: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if y.size == 0:
        raise HTTPException(status_code=422, detail="Decoded audio is empty")

    try:
        results = analyze_audio(y, sr)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return results