import streamlit as st
import whisper
import tempfile
import os
from audiorecorder import audiorecorder
from io import BytesIO
import sys
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import torch



st.set_page_config(page_title="Meeting Transcriber", layout="centered")
st.title("🎙️ Meeting Transcriber & Summarizer")
st.write("Upload or record meeting audio, get a transcript, then a summary.")

# 1) Whisper model (cached so it only loads once)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# 2) Summarization model + tokenizer (also cached)
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("Shaelois/MeetingScript")
    model = AutoModelForSeq2SeqLM.from_pretrained("Shaelois/MeetingScript")
    return tokenizer, model

summ_tokenizer, summ_model = load_summarizer()
# device for summarization
device = "cuda" if torch.cuda.is_available() else "cpu"
summ_model.to(device)

def sliding_window_summarize(
    text: str,
    tokenizer,
    model,
    window_size: int = 4096,
    stride: int = 1024,
    chunk_summary_max_length: int = 1000,
    final_summary: bool = True
):
    """
    1) Tokenize entire transcript without truncation.
    2) Slide a window of `window_size` tokens, stepping by (window_size - stride).
    3) Decode each chunk and run `model.generate(...)`.
    4) Optionally concatenate chunk summaries and do one more pass.
    """
    # 1) tokenize full text
    tok = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = tok.input_ids[0]  # shape (total_len,)
    n_tokens = input_ids.size(0)

    chunk_summaries = []
    for start in range(0, n_tokens, window_size - stride):
        end = min(start + window_size, n_tokens)
        chunk_ids = input_ids[start:end].unsqueeze(0).to(device)
        chunk_mask = torch.ones_like(chunk_ids).to(device)

        # 2) generate summary for this chunk
        out = model.generate(
            chunk_ids,
            attention_mask=chunk_mask,
            num_beams=4,
            max_length=chunk_summary_max_length,
            min_length=500,
            early_stopping=True,
        )
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        chunk_summaries.append(summary)

        if end == n_tokens:
            break

    # 3) if you want a single unified summary, re-summarize the pieces
    if final_summary and len(chunk_summaries) > 1:
        merged = " ".join(chunk_summaries)
        m_tok = tokenizer(
            merged,
            return_tensors="pt",
            truncation=True,
            max_length=window_size
        ).to(device)
        final_ids = model.generate(
            m_tok.input_ids,
            attention_mask=m_tok.attention_mask,
            num_beams=4,
            max_length=chunk_summary_max_length,
            min_length=40,
            early_stopping=True,
        )
        return tokenizer.decode(final_ids[0], skip_special_tokens=True)

    # otherwise, return list of per‑chunk summaries joined by newlines
    return "\n\n".join(chunk_summaries)

def transcribe_and_summarize(audio_path: str):
    # 1) Whisper transcription
    whisper_out = whisper_model.transcribe(audio_path)
    transcript = whisper_out["text"]

    # 2) Sliding‑window summarization
    summary = sliding_window_summarize(
        transcript,
        tokenizer=summ_tokenizer,
        model=summ_model,
        window_size=4096,      # BART’s positional limit
        stride=1024,            # 75% overlap
        chunk_summary_max_length=1000,
        final_summary=True     # set False to get all chunk summaries
    )
    return transcript, summary

input_method = st.radio("Select Audio Input Method:", ("Upload Audio File", "Record Audio"))

if input_method == "Upload Audio File":
    st.markdown("### 📁 Upload your meeting audio below")
    uploaded_file = st.file_uploader(label="Upload", type=["mp3","wav","m4a","mp4"], label_visibility="collapsed")

    if uploaded_file:
        status = st.empty()
        status.info("▶️ Transcribing… please wait")
        # write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        transcript_text, summary_text = transcribe_and_summarize(tmp_path)
        os.remove(tmp_path)
        status.empty()

        # Display transcript
        st.subheader("📝 Transcript")
        st.text_area("Transcript", transcript_text, height=300, label_visibility="collapsed")
        st.download_button(
            "💾 Download Transcript as .txt",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )

        # Display summary
        st.subheader("🗒️ Summary")
        st.text_area("Summary", summary_text, height=200, label_visibility="collapsed")
        st.download_button(
            "💾 Download Summary as .txt",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain"
        )

elif input_method == "Record Audio":
    st.markdown("### 🎙️ Click the button below to record")
    audio_segment = audiorecorder()

    if audio_segment.duration_seconds > 0:
        # play back
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        wav_bytes = buffer.getvalue()
        st.audio(wav_bytes, format="audio/wav")

        status = st.empty()
        status.info("▶️ Transcribing… please wait")
        # save to temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        transcript_text, summary_text = transcribe_and_summarize(tmp_path)
        os.remove(tmp_path)
        status.empty()

        # Display transcript
        st.subheader("📝 Transcript")
        st.text_area("Transcript", transcript_text, height=300, label_visibility="collapsed")
        st.download_button(
            "💾 Download Transcript as .txt",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )

        # Display summary
        st.subheader("🗒️ Summary")
        st.text_area("Summary", summary_text, height=200, label_visibility="collapsed")
        st.download_button(
            "💾 Download Summary as .txt",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain"
        )
