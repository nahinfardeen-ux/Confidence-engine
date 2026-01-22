
import os
import pickle
import numpy as np
# Lazy imports for stability
import speech_recognition as sr
import streamlit as st
from audiorecorder import audiorecorder
from scipy.io.wavfile import write as wav_write
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Paths
AUDIO_MODEL_PATH = "models/audio_model.h5"
TEMP_AUDIO_PATH = "temp_input.wav"

# Constants
DANGER_WORDS = ['risk', 'loss', 'fail', 'problem', 'sorry', 'um', 'uh']
POSITIVE_WORDS = ['sure', 'confident', 'happy', 'great', 'excellent', 'amazing', 'solution', 'success', 'absolute', 'grow', 'profit', 'love', 'best', 'win']


@st.cache_resource
def load_models():
    """Load audio, custom text models, and word lists."""
    # Lazy imports to verify crash source
    import tensorflow as tf
    import json
    import pickle
    
    audio_model = None
    vectorizer = None
    text_model = None

    try:
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load audio model: {e}")

    # Load Custom Text Model (Bigram-aware)
    try:
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("models/text_model.pkl", "rb") as f:
            text_model = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load custom text models: {e}")
    
    # Load Word Lists
    try:
        with open('models/negative_words.json', 'r') as f:
            neg_words = set(json.load(f))
        with open('models/positive_words.json', 'r') as f:
            pos_words = set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback
        neg_words = set(['risk', 'loss', 'fail', 'problem', 'sorry', 'um', 'uh'])
        pos_words = set(['sure', 'confident', 'happy', 'great', 'excellent', 'amazing', 'solution', 'success'])
        st.warning("Could not load full word lists. Using fallbacks.")

    return audio_model, vectorizer, text_model, neg_words, pos_words

def analyze_audio_file(filepath, audio_model, vectorizer, text_model, neg_words, pos_words):
    """Analyze the audio file and return metrics."""
    import librosa
    import string
    
    confidence_score = 0.0
    sales_score = 0.0
    transcript = ""
    warning_msg = ""
    breakdown = {}

    # --- Step A: Audio Confidence ---
    if audio_model:
        try:
            # Load audio (3 seconds)
            y, sr_lib = librosa.load(filepath, duration=3, sr=22050)
            
            # Calculate Volume (Average Amplitude) before normalization
            volume_rms = np.sqrt(np.mean(y**2))
            
            # --- Silence/Stutter Detection ---
            # Split non-silent chunks
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            
            # Calculate non-silent duration excluding pauses
            non_silent_samples = 0
            for start, end in non_silent_intervals:
                non_silent_samples += (end - start)
            
            total_samples = len(y)
            if total_samples > 0:
                # Pause Ratio: Percentage of time spent in silence
                pause_ratio = 1.0 - (non_silent_samples / total_samples)
            else:
                pause_ratio = 0.0
            
            # Fluency Score: Relaxed penalty (0.5x penalty instad of 2x)
            fluency_score = max(0.0, 1.0 - (pause_ratio * 0.5))
            
            # Debug info
            print(f"Volume(RMS): {volume_rms:.3f}, Pause Ratio: {pause_ratio:.2f}, Fluency: {fluency_score:.2f}")

            # Normalize Audio (CRITICAL for accuracy across different mics)
            y = librosa.util.normalize(y)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr_lib, n_mfcc=40)
            base_score = 0.0
            
            if mfccs.shape[1] > 0:
                 mfccs_mean = np.mean(mfccs.T, axis=0)
                 
                 # Reshape: (1, 40, 1)
                 input_data = np.expand_dims(mfccs_mean, axis=0) 
                 input_data = np.expand_dims(input_data, axis=-1) 
                 
                 # Predict (Calm=1, Fearful=0)
                 pred_probs = audio_model.predict(input_data, verbose=0)
                 base_score = float(pred_probs[0][1])
            
            # --- Hybrid Final Score (Step 1: Audio + Fluency) ---
            # 50% Model (Tone) + 50% Fluency (Pacing)
            raw_audio_score = (base_score * 0.5) + (fluency_score * 0.5)
            
            # Volume Bonus: If speaking loudly (RMS > 0.05), add 10%
            if volume_rms > 0.05: 
                 raw_audio_score += 0.10
            
            confidence_score = raw_audio_score

        except Exception as e:
            st.error(f"Error in Audio Analysis: {e}")

    # --- Step B: Transcription ---
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filepath) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        transcript = "(Unintelligible)"
    except Exception as e:
        st.error(f"Error in Transcription: {e}")

    # --- Step C: Sales Sentiment (Custom Model) ---
    if transcript and transcript != "(Unintelligible)" and vectorizer and text_model:
        try:
            # Transform text
            text_vec = vectorizer.transform([transcript])
            # Predict probabilities [Negative, Positive]
            probs = text_model.predict_proba(text_vec)
            # sales_score is probability of being Positive (index 1)
            sales_score = float(probs[0][1])
        except Exception as e:
            st.error(f"Error in Text Analysis: {e}")

    # --- Step D: Final "Smart" Confidence Calculation ---
    
    # Filter ambiguous intensifiers from positive words to avoid "High Risk" confusion
    ambiguous_words = {'high', 'significant', 'major', 'extreme', 'huge', 'big', 'large', 'increase'}
    pos_words = pos_words - ambiguous_words

    # 1. Danger/Positive Word Logic
    neg_count = 0
    pos_count = 0
    neg_hits = []
    pos_hits = []
    
    if transcript:
        words = transcript.lower().split()
        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word in neg_words:
                neg_count += 1
                neg_hits.append(clean_word)
            if clean_word in pos_words:
                pos_count += 1
                pos_hits.append(clean_word)
            
    # 2. Phrase Logic (Specific Grammar/Context Penalties)
    phrase_penalty = 0.0
    critical_phrases = ["have no idea", "make loss"]
    found_phrases = []
    
    if transcript:
        transcript_lower = transcript.lower()
        for phrase in critical_phrases:
            if phrase in transcript_lower:
                phrase_penalty += 0.10
                found_phrases.append(phrase)
    
    # Calculate Impact
    penalty_amount = min(0.30, neg_count * 0.05) 
    bonus_amount = min(0.20, pos_count * 0.05)
    
    # 3. New Formula: 50/50 Split (Audio/Content)
    base_confidence = confidence_score # Raw audio score (Tone + Fluency)
    
    real_confidence = (base_confidence * 0.50) + (sales_score * 0.50)
    
    # 4. Apply Modifiers
    final_confidence = real_confidence - penalty_amount - phrase_penalty + bonus_amount
    
    # 5. Safety Cap Logic (The Veto)
    is_capped = False
    if sales_score < 0.40:
        final_confidence = min(final_confidence, 0.45)
        is_capped = True
    
    # Clamp
    final_confidence = min(0.99, max(0.0, final_confidence))
    
    # Construct Reasoning Breakdown
    breakdown = {
        "Audio Tone Score": f"{base_score:.2%}",
        "Pacing/Fluency Score": f"{fluency_score:.2%}",
        "Volume Bonus": "+10%" if volume_rms > 0.05 else "0%",
        "Silence Ratio": f"{pause_ratio:.1%}",
        "Sentiment Score (VADER)": f"{sales_score:.2%}",
        "Negative Words Found": f"{neg_count} ({', '.join(set(neg_hits))})",
        "Bad Phrases Found": f"{', '.join(found_phrases)}" if found_phrases else "None",
        "Positive Words Found": f"{pos_count} ({', '.join(set(pos_hits))})",
        "Word Penalty": f"-{penalty_amount:.0%}",
        "Phrase Penalty": f"-{phrase_penalty:.0%}",
        "Bonus Applied": f"+{bonus_amount:.0%}",
        "Final Calculation": f"({base_confidence:.2f} * 0.5) + ({sales_score:.2f} * 0.5) - Penalties + Bonuses",
        "Score Capped?": "YES (Content < 40%)" if is_capped else "No"
    }

    if pause_ratio > 0.40: 
        if warning_msg: warning_msg += "\n"
        warning_msg += "⚠️ High Stutter/Pauses Detected."
        
    return final_confidence, sales_score, transcript, warning_msg, breakdown, is_capped

# --- Streamlit UI ---

st.set_page_config(page_title="Confidence Engine", page_icon=None, layout="centered")

# Custom CSS for Modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #F8FAFC;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(120deg, #2563EB 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: #64748B;
        margin-bottom: 3rem;
        text-align: center;
        font-weight: 400;
    }
    
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        margin-bottom: 1.5rem;
        border: 1px solid #F1F5F9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.025);
    }
    
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .highlight-red { 
        color: #EF4444; 
        font-weight: 600; 
        background-color: #FEF2F2;
        padding: 0 4px;
        border-radius: 4px;
    }
    .highlight-green { 
        color: #10B981; 
        font-weight: 600; 
        background-color: #ECFDF5;
        padding: 0 4px;
        border-radius: 4px;
    }
    
    /* Button refinement if accessed via st classes, hard to target directly without component IDs but global stButton helps */
    div.stButton > button {
        background-color: #0F172A;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #334155;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Confidence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Record your pitch to analyze confidence and sentiment.</div>', unsafe_allow_html=True)

# Load models
audio_model, vectorizer, text_model, neg_words, pos_words = load_models()

# Recorder
# Use a custom container for the recorder to style it, but avoid double visuals
with st.container():
    # Just output the recorder, maybe style the container via CSS selector if needed
    # The previous attempt put an empty card above the button presumably
    audio = audiorecorder("Click to Record", "Recording...")

if len(audio) > 0:
    # Playback the recorded audio
    # Just render the audio player directly without HTML wrapper to avoid layout issues
    st.audio(audio.export().read())
    
    # Save temporarily
    audio.export(TEMP_AUDIO_PATH, format="wav")

    # Add a Submit Button
    if st.button("Analyze Pitch"):
        with st.spinner("Analyzing..."):
            conf_score, sale_score, transcript_text, warning, breakdown, is_capped = analyze_audio_file(
                TEMP_AUDIO_PATH, audio_model, vectorizer, text_model, neg_words, pos_words
            )
            
            # Display Results Area
            if is_capped:
                st.error("Score Capped: Content is too weak to be confident.")

            if warning:
                st.warning(warning)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{conf_score:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf_score)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Sales Score</div>
                    <div class="metric-value">{sale_score:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(sale_score)
            
            st.markdown("### Transcript")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Smart Highlighting with Negative Dominance (Lookahead)
            if transcript_text and transcript_text != "(Unintelligible)":
                words = transcript_text.split()
                highlighted_html = []
                
                # Filter set for loop to avoid ambiguous words in highlighting too?
                ambiguous_words = {'high', 'significant', 'major', 'extreme', 'huge', 'big', 'large', 'increase'}
                display_pos_words = pos_words - ambiguous_words
                
                i = 0
                while i < len(words):
                    word = words[i]
                    import string
                    clean_word = word.strip(string.punctuation).lower()
                    
                    next_clean_word = None
                    if i + 1 < len(words):
                        next_clean_word = words[i+1].strip(string.punctuation).lower()
                    
                    # Negative Dominance Check
                    # If current is positive (e.g. 'High') and next is negative (e.g. 'Risk')
                    # We skip highlighting the current positive word.
                    if (clean_word in display_pos_words) and (next_clean_word in neg_words):
                        # Treat as neutral
                        highlighted_html.append(word)
                    elif clean_word in neg_words:
                        highlighted_html.append(f"<span class='highlight-red'>{word}</span>")
                    elif clean_word in display_pos_words:
                        highlighted_html.append(f"<span class='highlight-green'>{word}</span>")
                    else:
                        highlighted_html.append(word)
                    
                    i += 1
                
                final_html = " ".join(highlighted_html)
                st.markdown(final_html, unsafe_allow_html=True)
            else:
                st.info(transcript_text if transcript_text else "No speech detected.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Detailed Scoring Logic"):
                st.write("**Why did I get this score?**")
                
                b_col1, b_col2 = st.columns(2)
                
                with b_col1:
                    st.markdown("#### Audio Analysis")
                    st.write(f"**Tone:** {breakdown['Audio Tone Score']}")
                    st.write(f"**Fluency:** {breakdown['Pacing/Fluency Score']}")
                    st.write(f"**Volume:** {breakdown['Volume Bonus']}")
                    st.write(f"**Silence:** {breakdown['Silence Ratio']}")
                
                with b_col2:
                    st.markdown("#### Content Analysis")
                    st.write(f"**Sentiment (Custom Model):** {breakdown['Sentiment Score (VADER)']}")
                    st.write(f"**Phrases:** {breakdown['Phrase Penalty']}")
                    st.write(f"**Words:** {breakdown['Word Penalty']}")
                    st.write(f"**Bonus:** {breakdown['Bonus Applied']}")
                
                st.markdown("---")
                if is_capped:
                    st.error(f"**CAP APPLIED:** {breakdown['Score Capped?']}")
                st.caption(f"Final: {breakdown['Final Calculation']}")
    else:
        st.markdown('</div>', unsafe_allow_html=True) # Close recorder card if button not clicked yet logic (wait, button click resets render loop)
        # Actually logic for closing card needs to be clean.
        # Streamlit re-runs script on interaction.
        pass # The card close above in the 'if len > 0' block handles the audio player card.
else:
    # Close recorder card if no audio
    # The 'with st.container()' closed it? No, st.markdown raw html needs manual matching if strict, 
    # but here I opened card, put recorder, closed card. 
    # Wait, in lines 206-208 I opened and did not close?
    # Correcting: The recorder block
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    # audio = ...
    # st.markdown('</div>', unsafe_allow_html=True)
    pass

