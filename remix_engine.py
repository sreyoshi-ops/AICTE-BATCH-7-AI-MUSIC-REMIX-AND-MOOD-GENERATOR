import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter


# ----------------------------
# Utility Filters
# ----------------------------
def butter_filter(data, cutoff, sr, btype='low', order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)


# ----------------------------
# Bass Boost
# ----------------------------
def bass_boost(audio, sr, gain=1.5, cutoff=150):
    low_freq = butter_filter(audio, cutoff, sr, btype='low')
    return audio + gain * low_freq


# ----------------------------
# Echo with Feedback
# ----------------------------
def add_echo(audio, sr, delay_sec=0.3, decay=0.5, feedback=0.4):
    delay_samples = int(delay_sec * sr)
    echo_audio = np.copy(audio)

    for i in range(delay_samples, len(audio)):
        echo_audio[i] += decay * echo_audio[i - delay_samples] * feedback

    return echo_audio


# ----------------------------
# Reverb (Simple Convolution)
# ----------------------------
def add_reverb(audio, sr, reverb_strength=0.3):
    kernel_size = int(0.03 * sr)
    reverb_kernel = np.random.randn(kernel_size)
    reverb_kernel *= reverb_strength
    reverb_audio = np.convolve(audio, reverb_kernel, mode='same')
    return audio + reverb_audio


# ----------------------------
# Fade In / Fade Out (Safe)
# ----------------------------
def add_fade(audio, sr, fade_duration=2):
    fade_samples = min(int(fade_duration * sr), len(audio) // 2)

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    return audio


# ----------------------------
# Beat Drop Effect
# ----------------------------
def beat_drop(audio, sr, drop_time=5, drop_duration=1):
    start = int(drop_time * sr)
    end = min(start + int(drop_duration * sr), len(audio))
    audio[start:end] *= 0.1
    return audio


# ----------------------------
# Stereo Widening
# ----------------------------
def stereo_widen(audio):
    if len(audio.shape) == 1:
        audio = np.vstack([audio, audio])

    left = audio[0] * 1.1
    right = audio[1] * 0.9

    return np.vstack([left, right])


# ----------------------------
# MAIN REMIX FUNCTION
# ----------------------------
def remix_song(
        input_file,
        output_file,
        speed=1.2,
        pitch_shift=2,
        bass_gain=1.4,
        reverb_strength=0.2,
        echo_delay=0.25,
        echo_decay=0.6
):

    print("🎵 Loading audio...")
    y, sr = librosa.load(input_file, sr=None)

    # Speed change
    print("⚡ Changing speed...")
    y = librosa.effects.time_stretch(y, rate=speed)

    # Pitch shift
    print("🎼 Shifting pitch...")
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    # Bass boost
    print("🔊 Boosting bass...")
    y = bass_boost(y, sr, gain=bass_gain)

    # Echo
    print("🌊 Adding echo...")
    y = add_echo(y, sr, delay_sec=echo_delay, decay=echo_decay)

    # Reverb
    print("🎧 Adding reverb...")
    y = add_reverb(y, sr, reverb_strength=reverb_strength)

    # Beat drop
    print("💥 Adding beat drop...")
    y = beat_drop(y, sr)

    # Fade in/out
    print("🎚 Adding fade effects...")
    y = add_fade(y, sr)

    # Normalize safely
    print("📊 Normalizing...")
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.95  # Prevent clipping

    # Convert to stereo
    y = stereo_widen(y)

    # Save (transpose because soundfile expects shape (N, channels))
    print("💾 Saving remixed track...")
    sf.write(output_file, y.T, sr)

    print("✅ Remix complete!")
    return output_file
