import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import random
import tempfile
import os
import time

from remix_engine import remix_song

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MoodMixly AI 🎵",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS & ASSETS
# -----------------------------------------------------------------------------
def load_custom_css():
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        /* ROOT VARIABLES */
        :root {
            --primary-color: #00F5FF;
            --secondary-color: #FF00FF;
            --bg-dark: #0a0a0a;
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-main: #FFFFFF;
            --text-sub: #B0B0B0;
        }

        /* GLOBAL STYLES */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
        }
        
        /* BACKGROUND ANIMATION */
        .stApp {
            background: linear-gradient(-45deg, #120c18, #001f2e, #1f001f, #000000);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* HEADERS */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }

        h1 {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            text-shadow: 0 0 30px rgba(0, 245, 255, 0.3);
        }

        /* CARDS & CONTAINERS */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0, 245, 255, 0.2);
            border-color: var(--primary-color);
        }

        /* CUSTOM BUTTONS */
        .stButton > button {
            background: linear-gradient(90deg, #00F5FF, #00BFFF);
            color: #000 !important;
            font-weight: 700 !important;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 245, 255, 0.4);
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(0, 245, 255, 0.6);
        }

        /* DOWNLOAD BUTTON */
        .stDownloadButton > button {
            background: linear-gradient(90deg, #FF00FF, #BE2493);
            color: #fff !important;
            border-radius: 12px;
        }

        /* METRIC CARDS */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: var(--primary-color) !important;
            text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(10px);
            border-right: 1px solid var(--glass-border);
        }

        /* SLIDERS */
        .stSlider > div > div > div > div {
            background-color: var(--primary-color) !important;
        }

        /* FOOTER */
        .footer {
            margin-top: 80px;
            text-align: center;
            color: var(--text-sub);
            font-size: 0.9rem;
            padding: 20px;
            border-top: 1px solid var(--glass-border);
        }
        
        /* WAVE ANIMATION */
        .wave-container {
            width: 100%;
            height: 100px;
            overflow: hidden;
            position: relative;
            margin: 40px 0;
            mask-image: linear-gradient(to right, transparent, black 10%, black 90%, transparent);
        }
        .wave-bar {
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(90deg, var(--primary-color), var(--primary-color) 4px, transparent 4px, transparent 12px);
            opacity: 0.5;
            animation: moveWave 2s linear infinite;
        }
        .wave-bar.second {
            background: repeating-linear-gradient(90deg, var(--secondary-color), var(--secondary-color) 4px, transparent 4px, transparent 12px);
            position: absolute;
            top: 0;
            left: 0;
            animation: moveWave 3s linear infinite reverse;
            opacity: 0.3;
        }
        @keyframes moveWave {
            0% { transform: translateX(0); }
            100% { transform: translateX(50px); }
        }

        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & DEFINITIONS
# -----------------------------------------------------------------------------

MOODS = {
    "happy": {"base": 440, "bpm": 120, "icon": "😄"},
    "sad": {"base": 220, "bpm": 60, "icon": "😢"},
    "energetic": {"base": 660, "bpm": 140, "icon": "⚡"},
    "calm": {"base": 330, "bpm": 70, "icon": "🧘"},
    "romantic": {"base": 350, "bpm": 75, "icon": "💖"},
    "dark": {"base": 180, "bpm": 65, "icon": "🦇"},
    "lofi": {"base": 300, "bpm": 85, "icon": "☕"},
    "epic": {"base": 500, "bpm": 110, "icon": "⚔️"},
    "chill": {"base": 280, "bpm": 90, "icon": "🧊"},
    "focus": {"base": 400, "bpm": 100, "icon": "🧠"},
}

def sine_wave(freq, t, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t)

def adsr_envelope(signal, sr, attack=0.1, decay=0.2, sustain=0.7, release=0.3):
    length = len(signal)
    env = np.zeros(length)
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = length - (a + d + r)
    s = max(0, s) # Safety check
    
    # Adjust envelope parts if total length is too short
    if a + d + r > length:
        # Simplified fallback for very short clips
        return signal * np.linspace(1, 0, length)

    env[:a] = np.linspace(0, 1, a)
    env[a:a+d] = np.linspace(1, sustain, d)
    env[a+d:a+d+s] = sustain
    env[a+d+s:] = np.linspace(sustain, 0, r)
    return signal * env

def generate_drum_beat(t, sr, bpm=120):
    beat_interval = 60 / bpm
    drum = np.zeros_like(t)
    num_beats = int(t[-1] / beat_interval)
    
    for i in range(num_beats):
        start = int(i * beat_interval * sr)
        if start < len(drum):
            # Simple kick/snare synthesis
            end = min(start + 500, len(drum))
            drum[start:end] += np.random.randn(end-start) * 0.5 * np.exp(-np.linspace(0, 5, end-start))
            
    return drum

def generate_mood_music(output_file, mood="happy", duration=8, sr=22050):
    base_freq = MOODS[mood]["base"]
    bpm = MOODS[mood]["bpm"]

    t = np.linspace(0, duration, int(sr * duration))
    
    # Simple synthesis logic
    melody = (
        sine_wave(base_freq, t, 0.3) +
        sine_wave(base_freq * 1.5, t, 0.2)
    )
    bass = sine_wave(base_freq / 2, t, 0.25)
    drums = generate_drum_beat(t, sr, bpm)

    music = melody + bass + drums
    music = adsr_envelope(music, sr)
    
    # Normalize
    max_val = np.max(np.abs(music))
    if max_val > 0:
        music = music / max_val * 0.9

    # Stereo
    stereo_music = np.vstack((music, music * 0.95)).T
    sf.write(output_file, stereo_music, sr)
    return output_file


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
     theme_mode = st.selectbox("🎨 Interface Theme", ["Cyberpunk", "Minimal Dark", "Glass"])
     quality = st.select_slider("🔊 Render Quality", options=["Low", "Medium", "High", "Ultra"])
     st.markdown("---")
     st.markdown("### 🚀 About")
     st.info("MoodMixly uses advanced DSP to remix and generate audio in real-time. Built for creators.")

     st.markdown("## ⚙️ How It Works")

     st.markdown("""
    1️⃣ Upload your audio file (MP3 or WAV).  
    2️⃣ Adjust remix controls like speed, pitch, bass, reverb, and echo.  
    3️⃣ Click **IGNITE REMIX ENGINE** to process your track in real-time using advanced DSP.  
    4️⃣ Preview and download your newly remixed audio instantly.
""")
    
# ---------------------------
# THEME STYLING
# ---------------------------

if theme_mode == "Glass":
    st.markdown("""
        <style>
        /* GLASS THEME - VIBRANT BLUE */
        .stApp {
            background: linear-gradient(135deg, #000428, #004e92, #000428);
            background-size: 200% 200%;
            animation: gradientLink 15s ease infinite;
            color: #ffffff;
        }

        .block-container {
            background: rgba(0, 78, 146, 0.15);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border-radius: 24px;
            border: 1px solid rgba(0, 245, 255, 0.1);
            padding: 2.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 245, 255, 0.2);
        }
        
        .glass-card:hover {
            box-shadow: 0 0 25px rgba(0, 245, 255, 0.4);
            border-color: #00F5FF;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white !important;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
        }

        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0, 198, 255, 0.6);
            background: linear-gradient(90deg, #0072ff, #00c6ff);
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background-color: #00F5FF !important;
            box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(0, 4, 40, 0.85);
            border-right: 1px solid rgba(0, 245, 255, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)


elif theme_mode == "Minimal Dark":
    st.markdown("""
        <style>
        /* LIGHT NEON THEME */
        .stApp {
            background-color: #050505;
            color: #E0E0E0;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
        }

        .block-container {
            background: #0a0a0a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.05);
        }
        
        .glass-card {
            background-color: #111;
            border: 1px solid #333;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            border-color: #00FFFF;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
            transform: translateY(-2px);
        }

        /* Buttons - Neon Gradients */
        .stButton>button {
            background: linear-gradient(45deg, #00FFFF, #39FF14);
            color: #000000 !important;
            border-radius: 8px;
            border: none;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 12px 28px;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
        }

        .stButton>button:hover {
            background: linear-gradient(45deg, #39FF14, #00FFFF);
            transform: scale(1.05);
            box-shadow: 0 0 25px rgba(57, 255, 20, 0.6);
        }
        
        /* Secondary/Download Buttons */
        .stDownloadButton > button {
             background: transparent !important;
             border: 2px solid #FF00FF !important;
             color: #FF00FF !important;
             box-shadow: 0 0 5px rgba(255, 0, 255, 0.2);
        }
        .stDownloadButton > button:hover {
             background: #FF00FF !important;
             color: #000 !important;
             box-shadow: 0 0 20px rgba(255, 0, 255, 0.6);
        }

        /* Active Widget Colors */
        .stSlider > div > div > div > div {
            background-color: #00FFFF !important;
            box-shadow: 0 0 10px #00FFFF;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #222;
        }
        
        /* Inputs/Selectboxes */
        .stSelectbox > div > div {
            background-color: #111;
            color: white;
            border-color: #333;
        }
        </style>
    """, unsafe_allow_html=True)


        

    
# -----------------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------------
col_hero_1, col_hero_2 = st.columns([2, 1])

with col_hero_1:
    st.markdown('<h1>MOOD<span style="color:#FF00FF">MIXLY</span> AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; opacity: 0.8;">The ultimate AI-powered audio remixing station. Shape your sound, define your vibe.</p>', unsafe_allow_html=True)

with col_hero_2:
    # A placeholder for a cool visual or logo if desired, or just empty spacing
    pass

st.markdown('<div class="wave-container"><div class="wave-bar"></div><div class="wave-bar second"></div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
tab_remix, tab_gen, tab_stats = st.tabs(["🎛️ Remix Studio", "✨ Mood Generator", "📊 Analytics"])

# ------------------------------------
# TAB 1: REMIX STUDIO
# ------------------------------------
with tab_remix:
    st.markdown('<div class="glass-card"><h3>📂 Import Track</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your MP3 or WAV here", type=["mp3", "wav"])

    if uploaded_file:
        st.markdown("### 🎚️ Audio Controls")
        
        # Audio Controls Layout
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**⏱️ Tempo & Pitch**")
            speed = st.slider("Speed", 0.5, 2.0, 1.2, 0.1)
            pitch_shift = st.slider("Pitch (semitones)", -12, 12, 2)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🔊 Dynamics**")
            bass_gain = st.slider("Bass Boost", 1.0, 3.0, 1.4)
            reverb_strength = st.slider("Reverb", 0.0, 1.0, 0.2)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🌊 Echo / Delay**")
            echo_delay = st.slider("Delay (sec)", 0.1, 1.0, 0.25)
            echo_decay = st.slider("Decay", 0.1, 1.0, 0.6)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process Button
        if st.button("🚀 IGNITE REMIX ENGINE"):
            with st.spinner("🎧 Resynthesizing audio streams..."):
                # Save input
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
                    tmp_input.write(uploaded_file.read())
                    input_path = tmp_input.name
                
                # Output path
                with tempfile.NamedTemporaryFile(delete=False, suffix="_remix.wav") as tmp_output:
                    output_path = tmp_output.name

                # Processing
                remix_song(
                    input_path, output_path, speed, pitch_shift, bass_gain, 
                    reverb_strength, echo_delay, echo_decay
                )
                
                time.sleep(1) # Fake processing feel needed? Maybe not, but let's just show Spinner clearly
                
                # Render Result
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    st.success("✅ Remix Generated!")
                    st.audio(output_path, format="audio/wav")
                
                with res_col2:
                    with open(output_path, "rb") as f:
                        btn = st.download_button(
                            label="⬇️ Download Your Masterpiece",
                            data=f,
                            file_name="remixed_track.wav",
                            mime="audio/wav"
                        )
    else:
        st.info("👆 Upload a song to unlock the studio controls.")

# ------------------------------------
# TAB 2: MOOD GENERATOR
# ------------------------------------
with tab_gen:
    st.markdown('<div class="glass-card"><h3>✨ Generate AI Soundscapes</h3><p>Select a vibe and let the AI compose for you.</p></div>', unsafe_allow_html=True)
    
    # Mood Selection as Pills/Columns
    mood_list = list(MOODS.keys())
    selected_mood = st.selectbox("Choose Vibe", mood_list, format_func=lambda x: f"{MOODS[x]['icon']} {x.title()}")
    
    duration = st.slider("Track Duration (seconds)", 3, 30, 8)
    
    if st.button("🎹 Generate Mood Track"):
        with st.spinner("🤖 Composing original melody..."):
            file_path = "generated_music.wav"
            generate_mood_music(file_path, selected_mood, duration)
            time.sleep(0.5)
            
            st.balloons()
            st.markdown(f"### Now Playing: {MOODS[selected_mood]['icon']} {selected_mood.title()} Vibes")
            
            p1, p2 = st.columns([3, 1])
            with p1:
                st.audio(file_path, format="audio/wav")
            with p2:
                 with open(file_path, "rb") as f:
                    st.download_button("⬇ Save Track", f, file_name="mood_track.wav", mime="audio/wav")

# ------------------------------------
# TAB 3: ANALYTICS & PLANS
# ------------------------------------
with tab_stats:
    st.markdown("### 📈 Creator Dash")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tracks Remixed", "1,248", "+12%")
    m2.metric("Hours Streamed", "843", "+5%")
    m3.metric("Followers", "4.2K", "+84")
    m4.metric("Avg BPM", "128", "High Energy")
    
    st.markdown("### 💎 Go Pro")
    
    plan_col1, plan_col2 = st.columns(2)
    with plan_col1:
        st.markdown("""
            <div class="glass-card">
                <h3>Starter</h3>
                <h1 style="color:white; text-shadow:none;">Free</h1>
                <ul>
                    <li>Basic Effects</li>
                    <li>30sec Generation</li>
                    <li>MP3 Export</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with plan_col2:
        st.markdown("""
            <div class="glass-card" style="border: 1px solid var(--primary-color);">
                <h3>Studio Pro 👑</h3>
                <h1 style="color:var(--primary-color); text-shadow:none;">$19<span style="font-size:1rem">/mo</span></h1>
                <ul>
                    <li>Unlimited Remixes</li>
                    <li>Stem Separation</li>
                    <li>Lossless WAV Export</li>
                </ul>
                <br>
            </div>
        """, unsafe_allow_html=True)
        st.button("Upgrade to Pro")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="footer">
        <p>© 2026 MoodMixly AI • Crafted with 💜 & 🎧 using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
