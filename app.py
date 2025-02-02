import streamlit as st
import pyaudio
import requests
import wave
from io import BytesIO
import audioop
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
st.secrets["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Replace with your GROQ API key
CHUNK = 4096  # Increased chunk size for better performance
FORMAT = pyaudio.paInt24  # 24-bit audio format for higher quality
CHANNELS = 2  # Stereo audio for richer sound
RATE = 48000  # Higher sampling rate for better quality (48 kHz)
SILENCE_THRESHOLD = 550000  # Adjusted silence threshold for higher sensitivity
SILENCE_DURATION = 2  # Duration of silence (in seconds) to stop listening
MIN_SPEECH_DURATION = 3  # Minimum duration of speech (in seconds) before stopping
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio():
    """Function to record audio from the microphone."""
    # Open the microphone stream with higher quality settings
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=None)  # Use default microphone

    st.write("Listening... Start speaking.")

    frames = []
    silence_start_time = None
    speech_start_time = None

    # Placeholder for microphone strength
    strength_placeholder = st.empty()

    try:
        while True:
            # Read audio data from the microphone
            data = stream.read(CHUNK)
            frames.append(data)

            # Calculate the RMS (Root Mean Square) of the audio chunk
            rms = audioop.rms(data, 3)  # 3 is the sample width in bytes (24-bit audio)

            # Display microphone strength in real-time
            strength_placeholder.metric("Microphone Strength", f"{rms} dB")

            # Check if the audio energy is below the silence threshold
            if rms < SILENCE_THRESHOLD:
                if silence_start_time is None:
                    silence_start_time = time.time()  # Start timing silence
                else:
                    # If silence duration exceeds the threshold, stop listening
                    if time.time() - silence_start_time > SILENCE_DURATION:
                        # Ensure the user spoke for at least MIN_SPEECH_DURATION
                        if speech_start_time and (time.time() - speech_start_time > MIN_SPEECH_DURATION):
                            st.write("Silence detected. Stopping listening.")
                            break
            else:
                # Reset silence timer if speech is detected
                silence_start_time = None
                if speech_start_time is None:
                    speech_start_time = time.time()  # Start timing speech

    except KeyboardInterrupt:
        st.write("Stopping recording...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Convert the captured audio to a WAV file in memory
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        # Reset buffer position
        wav_buffer.seek(0)

        return wav_buffer

def transcribe_audio(wav_buffer):
    """Function to transcribe audio using the GROQ API."""
    # Send the audio to the GROQ API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    files = {
        "file": ("audio.wav", wav_buffer, "audio/wav"),
        "model": (None, "whisper-large-v3"),
    }

    response = requests.post(URL, headers=headers, files=files)

    if response.status_code == 200:
        transcription = response.json()
        return transcription["text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

def main():
    """Main function to run the Streamlit app."""
    st.title("🎤 Audio Transcription App")
    st.write("Click the button below to start recording and transcribing audio.")

    if st.button("Start Recording"):
        with st.spinner("Recording in progress..."):
            wav_buffer = record_audio()

        st.success("Recording complete! Transcribing audio...")

        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(wav_buffer)

        st.write("### Transcription:")
        st.write(transcription)

        # Clean up
        wav_buffer.close()

if __name__ == "__main__":
    main()
