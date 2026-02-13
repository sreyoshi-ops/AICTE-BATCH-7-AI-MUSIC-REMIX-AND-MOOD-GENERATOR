import numpy as np
import soundfile as sf


# -----------------------------
# Utility Functions
# -----------------------------

def sine_wave(freq, t, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t)


def adsr_envelope(signal, sr, attack=0.1, decay=0.2, sustain=0.7, release=0.3):
    length = len(signal)
    env = np.zeros(length)

    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = length - (a + d + r)

    env[:a] = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, sustain, d)
    env[a+d:a+d+s] = sustain
    env[a+d+s:] = np.linspace(sustain, 0, r)

    return signal * env


def generate_drum_beat(t, sr, bpm=120):
    beat_interval = 60 / bpm
    drum = np.zeros_like(t)

    for i in range(int(t[-1] / beat_interval)):
        start = int(i * beat_interval * sr)
        if start < len(drum):
            drum[start:start+200] += np.random.randn(200) * 0.3

    return drum


# -----------------------------
# Mood Settings
# -----------------------------

MOODS = {
    "happy": {"base": 440, "bpm": 120},
    "sad": {"base": 220, "bpm": 60},
    "energetic": {"base": 660, "bpm": 140},
    "calm": {"base": 330, "bpm": 70},
    "romantic": {"base": 350, "bpm": 75},
    "dark": {"base": 180, "bpm": 65},
    "lofi": {"base": 300, "bpm": 85},
    "epic": {"base": 500, "bpm": 110},
    "chill": {"base": 280, "bpm": 90},
    "focus": {"base": 400, "bpm": 100},
    "uplifting": {"base": 480, "bpm": 125},
    "mysterious": {"base": 210, "bpm": 80}
}


# -----------------------------
# MAIN FUNCTION
# -----------------------------

def generate_mood_music(output_file,
                        mood="happy",
                        duration=8,
                        sr=22050):

    if mood not in MOODS:
        mood = "calm"

    base_freq = MOODS[mood]["base"]
    bpm = MOODS[mood]["bpm"]

    t = np.linspace(0, duration, int(sr * duration))

    # 🎵 Melody Layer (major/minor intervals)
    melody = (
        sine_wave(base_freq, t, 0.3) +
        sine_wave(base_freq * 1.25, t, 0.2) +
        sine_wave(base_freq * 1.5, t, 0.2)
    )

    # 🎹 Pad Layer (soft background)
    pad = sine_wave(base_freq / 2, t, 0.15)

    # 🔊 Bass Layer
    bass = sine_wave(base_freq / 4, t, 0.25)

    # 🥁 Drum Beat
    drums = generate_drum_beat(t, sr, bpm)

    # Combine all layers
    music = melody + pad + bass + drums

    # Apply envelope (smooth fade)
    music = adsr_envelope(music, sr)

    # Normalize safely
    max_val = np.max(np.abs(music))
    if max_val > 0:
        music = music / max_val * 0.9

    # Stereo effect
    stereo_music = np.vstack((music, music * 0.95)).T

    sf.write(output_file, stereo_music, sr)

    return output_file
