#!/bin/bash

# Fix for potential macOS OpenMP/TensorFlow runtime crashes
export KMP_DUPLICATE_LIB_OK=TRUE
export TF_ENABLE_ONEDNN_OPTS=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Activate virtual environment
source venv/bin/activate

# install ffmpeg if missing (pydub needs it) - straightforward check
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg could not be found. Detailed audio processing might fail."
    echo "If on Mac, try: brew install ffmpeg"
fi

# Run Streamlit
streamlit run src/app.py
