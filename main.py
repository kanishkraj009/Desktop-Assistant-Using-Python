import pyttsx3
from flask import Flask, jsonify, request
import datetime
import speech_recognition as sr
import pyaudio
import wikipedia
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY_HERE")  # Replace with your actual API key
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')


def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening...')
        r.pause_threshold = 2
        r.energy_threshold = 200
        audio = r.listen(source)
    
    try:
        print('Recognizing...')
        query = r.recognize_google(audio, language='en-in')
        print(f"User Said:{query}\n")
        return query.lower()
    except Exception as e:
        print('Say That Again Please...')
        return ''


def greet():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning Sir")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon Sir")
    else:
        speak("Good Evening Sir")

if __name__ == '__main__':
    greet()
    while True:

        query = voice_input()
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(query)
        out = response.text.replace('*', '')
        print(f"Machine Said:{out}")
        speak(out)

        if 'wikipedia' in query:
            speak('Searching on the web...')
            query = query.replace('wikipedia', '')
            try:
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia:")
                speak(results)
            except wikipedia.DisambiguationError as e:
                speak("The topic is ambiguous. Please be more specific.")
            except wikipedia.PageError:
                speak("Sorry, I could not find any information on that topic.")
