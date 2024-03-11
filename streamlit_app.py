import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import transformers
from transformers import pipeline
from langchain import HuggingFaceHub, HuggingFacePipeline
from transformers import pipeline, Conversation
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from pydub import AudioSegment
import requests
import io

def split_audio(audio_bytes, chunk_length_ms):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunks.append(chunk)

    return chunks

def transcribe_audio_with_deepgram(audio_bytes, api_key):
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3"
    }
    response = requests.post(url, headers=headers, data=audio_bytes)
    return response.json()


def generate_advice(last_lines):
    template = """Given this conversation: {last_lines}:
     1)Analyze how much every partner is involved in the conversation
     2)Give advice on how to improve this conversation
     3)In how many seconds should I give an answer?
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm = HuggingFacePipeline.from_model_id(huggingfacehub_api_token = "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",
        model_id="databricks/dolly-v2-3b",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 80},
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    analysis = llm_chain.run(last_lines)
    chatbot = pipeline(model="facebook/blenderbot-400M-distill")

    conversation = Conversation(f"Give me advice on how to continue communication. Make sure you use this recommendations: {analysis}")

    conversation.add_message({"role": "user", "content": "What should I say to my partner?"})
    return chatbot(conversation.messages[-1]["content"])



st.title("Советник по телефонным разговорам")

audio_file = st.file_uploader("Загрузите аудиозапись диалога", type=['mp3', 'wav'])

if audio_file is not None:
    # Читаем аудиофайл
    audio_bytes = audio_file.getvalue()

    # Предварительная обработка и транскрипция
    chunk_length_ms = 10000  # Примерная длина фрагмента в мс
    chunks = split_audio(audio_bytes, chunk_length_ms)
    api_key = "f818507c62d425b0f00c774e1d3e276f75b3de0c" 

    transcripts = []
    for chunk in chunks:
        audio_bytes = io.BytesIO()
        chunk.export(audio_bytes, format="mp3")
        transcription = transcribe_audio_with_deepgram(audio_bytes.getvalue(), api_key)
        transcripts.append(transcription['results']['channels'][0]['alternatives'][0]['transcript'])

    if transcripts:
        st.write("Транскрипция диалога:")
        for t in transcripts:
            st.write(t)


        last_lines = transcripts[-2:] 
        advice = generate_advice(last_lines)
        st.write("Совет по ведению диалога:")
        st.write(advice)
    else:
        st.write("Не удалось распознать диалог.")
else:
    st.write("Не удалось распознать диалог.")
