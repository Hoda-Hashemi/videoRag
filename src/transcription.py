import whisper
import re

def transcribe_audio(audio_file, transcription_file, segments_file, chunks_file, clean_chunks_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcript = result["text"]
    
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    segments = []
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        segments.append((start, end, text))
    
    with open(segments_file, "w", encoding="utf-8") as f:
        for start, end, text in segments:
            f.write(f"[{start:.1f}s-{end:.1f}s]: {text}\n")
    
    text_chunks = []
    for start, end, text in segments:
        timestamp = f"{int(start):04d}s-{int(end):04d}s"
        text_chunks.append(f"[{timestamp}]: {text}")
    
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    
    clean_chunks = []
    for chunk in text_chunks:
        match = re.match(r"\[(.*?)\]:\s*(.*)", chunk)
        if match:
            clean_chunks.append(match.group(2).strip())
    
    with open(clean_chunks_file, "w", encoding="utf-8") as f:
        for chunk in clean_chunks:
            f.write(chunk + "\n")
    
    return clean_chunks

