import os
import streamlit as st
import tempfile
from speechbrain.inference.interfaces import foreign_class
import soundfile
import transformers
import speechbrain

st.set_page_config(layout="wide")

st.title("Audio Sentiment Analysis")
st.write("[ENGRO](https://engro.io/)")

st.sidebar.title("Description")
st.sidebar.write("The tool allows one to upload an audio file and perform sentiment analysis")

st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
upload_button = st.sidebar.button("Upload")

def perform_sentiment_analysis(audiofile):
  classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
  out_prob, score, index, text_lab = classifier.classify_file(audiofile)
  sentiment_label = ''
  if text_lab[0] == 'hap':
    sentiment_label = 'POSITIVE'
  elif text_lab[0] == 'neu':
    sentiment_label = 'NEUTRAL'
  else: sentiment_label = 'NEGATIVE'
  return sentiment_label

def main():
  if audio_file and upload_button:
    try:
      with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name  # get the path of the saved temp file
      
        sentiment_label = perform_sentiment_analysis(tmp_path)
        st.header("Sentiment Analysis")
        negative_icon = "👎"
        neutral_icon = "😐"
        positive_icon = "👍"

        if sentiment_label == "NEGATIVE":
            st.write(f"{negative_icon} Negative", unsafe_allow_html=True)
        else:
            st.empty()

        if sentiment_label == "NEUTRAL":
            st.write(f"{neutral_icon} Neutral", unsafe_allow_html=True)
        else:
            st.empty()

        if sentiment_label == "POSITIVE":
            st.write(f"{positive_icon} Positive", unsafe_allow_html=True)
        else:
            st.empty()
            
        os.unlink(tmp_path)

    except Exception as ex:
      st.error("Error occurred during audio transcription and sentiment analysis.")
      st.error(str(ex))
      traceback.print_exc()

if __name__ == "__main__": main()
