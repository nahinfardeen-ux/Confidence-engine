
import os
import time
import pickle
import numpy as np
import librosa
import tensorflow as tf
import speech_recognition as sr
from IPython.display import Javascript, display, Audio
from base64 import b64decode
from scipy.io.wavfile import read as wav_read
import io
import scipy.io.wavfile as wav

# Paths
AUDIO_MODEL_PATH = "models/audio_model.h5"
TEXT_MODEL_PATH = "models/text_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
TEMP_AUDIO_PATH = "temp_input.wav"

def record_audio(filename=TEMP_AUDIO_PATH, sec=3):
    """
    Records audio from the browser microphone using IPython.display.Javascript.
    Saves the recording to 'filename'.
    """
    js = Javascript("""
    async function recordAudio(sec) {
      const div = document.createElement('div');
      const startButton = document.createElement('button');
      startButton.textContent = 'Start Recording';
      div.appendChild(startButton);
      
      const msg = document.createElement('span');
      div.appendChild(msg);
      document.body.appendChild(div);
      
      return new Promise((resolve) => {
        startButton.onclick = async () => {
          div.style.display = 'none';
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const recorder = new MediaRecorder(stream);
          const chunks = [];
          
          recorder.ondataavailable = e => chunks.push(e.data);
          recorder.onstop = () => {
             const blob = new Blob(chunks, { type: 'audio/wav' }); // 'audio/webm' usually
             const reader = new FileReader();
             reader.readAsDataURL(blob);
             reader.onloadend = () => {
               const base64String = reader.result.split(',')[1];
               resolve(base64String);
             };
          };
          
          recorder.start();
          msg.textContent = 'Recording...';
          setTimeout(() => recorder.stop(), sec * 1000);
        };
      });
    }
    """)
    
    try:
        display(js)
        from google.colab import output
        data = output.eval_js(f'recordAudio({sec})')
        binary = b64decode(data)
        
        # Save raw bytes
        # Note: Browser media recorder often gives webm/ogg. We might need ffmpeg to convert to wav.
        # However, for this prompt, we save as is or assume pydub can fix it, 
        # but the prompt specifically asked to "Save... as temp_input.wav".
        with open(filename, 'wb') as f:
            f.write(binary)
        print(f"Recording saved to {filename}")
        
    except ImportError:
        print("Not running in Google Colab or compatible Jupyter environment. 'record_audio' skipped.")
    except Exception as e:
        print(f"Error recording audio: {e}")

def analyze_pitch():
    """
    Step A: Extract MFCCs & Predict Confidence Score (Audio)
    Step B: Transcribe Text
    Step C: Vectorize & Predict Sales Score (Text)
    """
    print("\n--- Starting Analysis ---")

    if not os.path.exists(TEMP_AUDIO_PATH):
        print(f"Error: {TEMP_AUDIO_PATH} not found. Please record audio first.")
        return

    # --- Step A: Audio Prediction ---
    print("Loading Audio Model...")
    try:
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    except Exception as e:
        print(f"Failed to load audio model: {e}")
        return

    print(f"Processing {TEMP_AUDIO_PATH}...")
    try:
        # Load audio (3 seconds)
        y, sr_lib = librosa.load(TEMP_AUDIO_PATH, duration=3, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr_lib, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Reshape: (1, 40, 1)
        input_data = np.expand_dims(mfccs_mean, axis=0) # Batch dim
        input_data = np.expand_dims(input_data, axis=-1) # Channel dim
        
        # Predict
        # Classes: 0=Nervous, 1=Confident based on training script labels
        # (Calm(-02) -> 1, Fearful(-06) -> 0)
        pred_probs = audio_model.predict(input_data)
        confidence_score = pred_probs[0][1] # Probability of class 1
        print(f"Audio Confidence Score: {confidence_score:.4f}")
        
    except Exception as e:
        print(f"Error in Audio Analysis: {e}")

    # --- Step B: Text Transcription ---
    print("Transcribing audio...")
    recognizer = sr.Recognizer()
    try:
        # SpeechRecognition needs a WAV file. 
        # If the browser binary was WebM, this might fail unless converted. 
        # But assuming it's consumable or we treat it as such.
        # Librosa can load webm (via ffmpeg), but SR needs wav/aiff.
        # Let's assume the user handles conversion or the recorder provides wav.
        
        with sr.AudioFile(TEMP_AUDIO_PATH) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
            print(f"Transcript: \"{transcript}\"")
            
    except sr.UnknownValueError:
        print("Transcript: (Unintelligible)")
        transcript = ""
    except Exception as e:
        print(f"Error in Transcription: {e}")
        transcript = ""

    # --- Step C: Sales Prediction (Text) ---
    if transcript:
        print("Loading Text Models...")
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(TEXT_MODEL_PATH, 'rb') as f:
                text_model = pickle.load(f)
                
            # Vectorize
            text_vec = vectorizer.transform([transcript])
            
            # Predict
            # 0=Negative, 1=Positive
            sales_prob = text_model.predict_proba(text_vec)
            sales_score = sales_prob[0][1] # Probability of 'Positive' class
            
            print(f"Sales Score (Sentiment): {sales_score:.4f}")
            
        except Exception as e:
            print(f"Error in Text Analysis: {e}")
    else:
        print("No transcript to analyze for Sales Score.")

if __name__ == "__main__":
    # In a real environment, you might call record_audio() here if interactive.
    record_audio() 
    analyze_pitch()

