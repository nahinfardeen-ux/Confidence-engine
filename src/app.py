
import os
import pickle
import numpy as np
import string
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

    # --- Step A: Audio Confidence (Pitch Stability) ---
    if audio_model:
        try:
            # Load audio (3 seconds)
            y, sr_lib = librosa.load(filepath, duration=3, sr=22050)
            
            # --- Pitch Stability Analysis ---
            # Extract Pitch ($f0$) using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            # Filter NaNs (unvoiced parts)
            f0_clean = f0[~np.isnan(f0)]
            
            pitch_score_modifier = 0.0
            pitch_std = 0.0
            pitch_status = "Unknown"
            
            if len(f0_clean) > 0:
                pitch_std = np.std(f0_clean)
                
                # Rule: High Variance (> 50Hz) -> Nervous/Shaky
                if pitch_std > 50:
                    pitch_score_modifier = -0.15
                    pitch_status = "Shaky (High Variance)"
                # Rule: Low Variance (< 20Hz approx, but staying strictly to 'Stable' bonus logic) -> Stable
                # Bonus if decent stability, let's say < 30Hz or just 'not shaky' with some floor? 
                # Request said: "If std_dev is Low (Stable), give +10% Bonus."
                # Let's define Low as < 25Hz
                elif pitch_std < 25:
                    pitch_score_modifier = 0.10
                    pitch_status = "Stable (Confident)"
                else:
                    pitch_status = "Neutral"
            
            # --- Silence/Stutter Detection ---
            non_silent_intervals = librosa.effects.split(y, top_db=20)
            non_silent_samples = 0
            for start, end in non_silent_intervals:
                non_silent_samples += (end - start)
            
            total_samples = len(y)
            if total_samples > 0:
                pause_ratio = 1.0 - (non_silent_samples / total_samples)
            else:
                pause_ratio = 0.0
            
            # Fluency Score
            fluency_score = max(0.0, 1.0 - (pause_ratio * 0.5))
            
            print(f"Pitch Std: {pitch_std:.2f}, Status: {pitch_status}, Pause Ratio: {pause_ratio:.2f}")

            # Normalize Audio
            y = librosa.util.normalize(y)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr_lib, n_mfcc=40)
            base_score = 0.0
            
            if mfccs.shape[1] > 0:
                 mfccs_mean = np.mean(mfccs.T, axis=0)
                 input_data = np.expand_dims(mfccs_mean, axis=0) 
                 input_data = np.expand_dims(input_data, axis=-1) 
                 
                 pred_probs = audio_model.predict(input_data, verbose=0)
                 base_score = float(pred_probs[0][1])
            
            # --- Hybrid Final Score (Step 1: Audio + Fluency) ---
            raw_audio_score = (base_score * 0.5) + (fluency_score * 0.5)
            
            # Apply Pitch Modifier directly to audio component
            raw_audio_score += pitch_score_modifier
            
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
            text_vec = vectorizer.transform([transcript])
            probs = text_model.predict_proba(text_vec)
            sales_score = float(probs[0][1])
        except Exception as e:
            st.error(f"Error in Text Analysis: {e}")

    # --- Step D: Final "Smart" Confidence Calculation ---
    
    # Filter ambiguous
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
            
    # 2. Phrase Logic
    phrase_penalty = 0.0
    critical_phrases = ["have no idea", "make loss"]
    found_phrases = []
    
    if transcript:
        transcript_lower = transcript.lower()
        for phrase in critical_phrases:
            if phrase in transcript_lower:
                phrase_penalty += 0.10
                found_phrases.append(phrase)
    
    # 3. Filler Word Logic
    filler_penalty = 0.0
    fillers = ['um', 'uh', 'ah', 'like', 'actually', 'basically', 'sort of']
    found_fillers = []
    
    if transcript:
        # Check counts
        t_low = transcript.lower()
        for fill in fillers:
            # We use word boundary checking or just simple count if single word?
            # User list has "sort of" (2 words).
            # Simple count is easiest for now.
            count = t_low.count(fill)
            if count > 0:
                filler_penalty += (0.02 * count) # 2% per filler
                found_fillers.append(f"{fill} ({count})")
    
    # Calculate Impact (kept for breakdown/interest)
    penalty_amount = min(0.30, neg_count * 0.05) 
    bonus_amount = min(0.20, pos_count * 0.05)
    
    # --- New "Natural" Formula (Weighted Mixing) ---
    # Formula: Final_Confidence = (Audio_Model_Prediction * 0.60) + (Fluency_Score * 0.40)
    # where Fluency_Score = 1.0 - Pause_Ratio
    
    fluency_component = max(0.0, 1.0 - pause_ratio)
    
    # We use the raw probability from the audio model (base_score)
    # base_score is 0.0 to 1.0 (Calm/Confident probability)
    
    final_confidence = (base_score * 0.60) + (fluency_component * 0.40)
    
    # Clamp just in case, though math should keep it 0-1
    final_confidence = min(0.99, max(0.0, final_confidence))
    
    # Construct Breakdown
    breakdown = {
        "Audio Tone Score (AI)": f"{base_score:.1%}",
        "Fluency Score (Flow)": f"{fluency_component:.1%}",
        "Weighted Audio (60%)": f"{base_score * 0.60:.2f}",
        "Weighted Fluency (40%)": f"{fluency_component * 0.40:.2f}",
        "Pitch Stability": f"{pitch_status} ({pitch_std:.1f}Hz)",
        "Sales/Sentiment Score": f"{sales_score:.1%}",
        "Filler Words Found": f"{len(found_fillers)}",
        "Negative Words": f"{neg_count}",
        "Positive Words": f"{pos_count}",
        "Final Score Calculation": f"({base_score:.2f} * 0.60) + ({fluency_component:.2f} * 0.40)"
    }

    if pause_ratio > 0.40: 
        if warning_msg: warning_msg += "\n"
        warning_msg += "⚠️ High Stutter/Pauses Detected."
        
    return final_confidence, sales_score, transcript, warning_msg, breakdown

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
            conf_score, sale_score, transcript_text, warning, breakdown = analyze_audio_file(
                TEMP_AUDIO_PATH, audio_model, vectorizer, text_model, neg_words, pos_words
            )
            
            # Display Results Area
            if warning:
                st.warning(warning)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{conf_score:.1%}</div>
                    <div style="font-size:0.8rem; color:#64748B; margin-top:5px;">
                        Tone ({breakdown['Weighted Audio (60%)']}) + Fluency ({breakdown['Weighted Fluency (40%)']})
                    </div>
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

            
            # Smart Highlighting (Phrase-First Priority)
            if transcript_text and transcript_text != "(Unintelligible)":
                import re
                import uuid
                
                # Placeholder system to avoid double-coloring
                replacements = {}
                
                def get_placeholder():
                    uid = str(uuid.uuid4()).replace("-", "")
                    return f"PH_{uid}"
                
                processed_text = transcript_text
                
                # --- Step 1: Negative Phrases (Priority) ---
                # A. Ambiguous Intensifiers + Negative Word (e.g. "High Risk")
                # B. Explicit Bad Phrases (e.g. "not good")
                
                ambiguous_list = ['high', 'significant', 'major', 'extreme', 'huge', 'big', 'large', 'increase']
                explicit_phrases = ["not good", "have no idea", "make loss", "to be honest", "sort of"]
                
                # 1A. Construct Regex for Ambiguous + Negative
                # Pattern: \b(high|significant...)\s+(\w+)\b
                ambig_pattern = r'\b(' + '|'.join(ambiguous_list) + r')\s+(\w+)\b'
                
                def phrase_sub(match):
                    full_phrase = match.group(0)
                    next_word = match.group(2).lower().strip(string.punctuation)
                    
                    # Check if the second word is negative
                    if next_word in neg_words:
                        ph = get_placeholder()
                        replacements[ph] = f"<span class='highlight-red'>{full_phrase}</span>"
                        return ph
                    return full_phrase

                processed_text = re.sub(ambig_pattern, phrase_sub, processed_text, flags=re.IGNORECASE)
                
                # 1B. Explicit Phrases
                for phr in explicit_phrases:
                    # Regex escape just in case
                    pattern = re.compile(re.escape(phr), re.IGNORECASE)
                    
                    def explicit_sub(match):
                        ph = get_placeholder()
                        replacements[ph] = f"<span class='highlight-red'>{match.group(0)}</span>"
                        return ph
                        
                    processed_text = pattern.sub(explicit_sub, processed_text)
                
                # --- Step 2: Positive Words (Remaining) ---
                # Filter ambiguous out of simple positive highlighting to avoid coloring "High" in "High Mountain" if we wanted?
                # But actually, prompt says "Find positive words...". 
                # Since "High Risk" is already replaced by placeholder, "High" won't match here if it was part of that phrase.
                # If "High" is alone, it stays. The prompt says "Override: Do NOT color 'High' green" for High Risk case.
                # We handled that. But what about "High Value"? That should be green.
                # So we simply match words now.
                
                def word_sub_pos(match):
                    word = match.group(0)
                    clean = word.lower().strip(string.punctuation)
                    if clean in pos_words:
                        ph = get_placeholder()
                        replacements[ph] = f"<span class='highlight-green'>{word}</span>"
                        return ph
                    return word
                
                # Iterate words to check sets (Regex for all pos words is too big, so we iterate tokens)
                # But iterating tokens destroys structure if we just join them back?
                # Regex sub with callback for \w+ works best to preserve spacing/punctuation.
                
                processed_text = re.sub(r'\b\w+\b', word_sub_pos, processed_text)
                
                # --- Step 3: Negative Words (Remaining) ---
                def word_sub_neg(match):
                    word = match.group(0)
                    clean = word.lower().strip(string.punctuation)
                    if clean in neg_words:
                        ph = get_placeholder()
                        replacements[ph] = f"<span class='highlight-red'>{word}</span>"
                        return ph
                    return word
                    
                processed_text = re.sub(r'\b\w+\b', word_sub_neg, processed_text)
                
                # --- Step 4: Restore Placeholders ---
                for ph, html in replacements.items():
                    processed_text = processed_text.replace(ph, html)
                
                st.markdown(f'<div class="card">{processed_text}</div>', unsafe_allow_html=True)
            else:
                st.info(transcript_text if transcript_text else "No speech detected.")

            with st.expander("Detailed Scoring Logic"):
                st.write("**Why did I get this score?**")
                
                b_col1, b_col2 = st.columns(2)
                
                with b_col1:
                    st.markdown("#### Audio Analysis")
                    st.write(f"**Tone (AI):** {breakdown['Audio Tone Score (AI)']}")
                    st.write(f"**Fluency:** {breakdown['Fluency Score (Flow)']}")
                    st.write(f"**Pitch Stability:** {breakdown['Pitch Stability']}")
                    st.caption(f"Score = ({breakdown['Audio Tone Score (AI)']} * 0.6) + ({breakdown['Fluency Score (Flow)']} * 0.4)")
                
                with b_col2:
                    st.markdown("#### Content Metrics (Info Only)")
                    st.write(f"**Sentiment Score:** {breakdown['Sales/Sentiment Score']}")
                    st.write(f"**Fillers:** {breakdown['Filler Words Found']}")
                    st.write(f"**Negative Words:** {breakdown['Negative Words']}")
                    st.write(f"**Positive Words:** {breakdown['Positive Words']}")
                
                st.markdown("---")
                st.caption(f"Final: {breakdown['Final Score Calculation']}")
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

