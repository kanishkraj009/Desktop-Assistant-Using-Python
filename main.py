import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import os
import cv2
import numpy as np
import threading
import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import google.generativeai as genai

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
genai.configure(api_key="YOUR API KEY")

# Context memory and custom response data
context_memory = []
custom_responses_file = "custom_responses.json"

# Load or initialize custom responses
if not os.path.exists(custom_responses_file):
    with open(custom_responses_file, 'w') as f:
        json.dump({}, f)

with open(custom_responses_file, 'r') as f:
    custom_responses = json.load(f)

# Speak function with interrupt capability
stop_speaking = False

def speak(audio):
    global stop_speaking
    stop_speaking = False
    engine.say(audio)
    engine.runAndWait()

# Voice input function
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 2
        r.energy_threshold = 200
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User Said: {query}\n")
            return query.lower()
        except sr.WaitTimeoutError:
            print("No input detected.")
            return ""
        except Exception:
            print("I didn't catch that. Can you say it again?")
            return ""

# Greet function
def greet():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")

# Intent classifier
def classify_intent(query):
    try:
        with open('intent_classifier.pkl', 'rb') as f:
            vectorizer, clf = pickle.load(f)
        query_vec = vectorizer.transform([query])
        intent = clf.predict(query_vec)[0]
        return intent
    except FileNotFoundError:
        print("Intent classifier model not found.")
        return "unknown"

# Train intent classifier with preloaded conversations
def train_intent_classifier():
    data = {
        "query": [
            "What's the weather today?",
            "Tell me about Python programming.",
            "Good morning!",
            "Stop the program.",
            "Create a story for me.",
            "What can you do?",
            "Tell me a joke.",
            "What's the time?",
            "Who are you?",
            "Search for Einstein on Wikipedia.",
            "Take a picture"
        ],
        "intent": [
            "weather",
            "generative_ai",
            "greeting",
            "exit",
            "generative_ai",
            "capabilities",
            "joke",
            "time",
            "identity",
            "wikipedia_search",
            "take_picture"
        ]
    }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(df['query'], df['intent'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = SVC(probability=True)
    clf.fit(X_train_vec, y_train)

    with open('intent_classifier.pkl', 'wb') as f:
        pickle.dump((vectorizer, clf), f)
    print("Intent classifier trained and saved.")

# Add to context memory
def add_to_context(query):
    if len(context_memory) > 5:
        context_memory.pop(0)
    context_memory.append(query)

# Save custom responses
def save_custom_responses():
    with open(custom_responses_file, 'w') as f:
        json.dump(custom_responses, f)

# Face Detection using OpenCV (Haar Cascade)
def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    detected_faces = []

    for (x, y, w, h) in faces:
        detected_faces.append(f"Face detected at position x:{x}, y:{y}, width:{w}, height:{h}")
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imwrite("detected_faces.jpg", image)  # Save image with detected faces
    return detected_faces

# Capture image function
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return None
    print("Capturing image...")
    ret, frame = cap.read()
    if ret:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Image saved at {image_path}")
    else:
        print("Failed to capture image.")
        return None
    cap.release()
    return image_path

# Main assistant loop
if __name__ == '__main__':
    train_intent_classifier()
    greet()

    while True:
        query = voice_input()

        if query:
            add_to_context(query)

            # Check for custom responses
            if query in custom_responses:
                speak(custom_responses[query])
                continue

            # Classify intent
            intent = classify_intent(query)
            print(f"Intent: {intent}")

            if intent == "greeting":
                speak("Hello! How can I assist you?")
            elif intent == "wikipedia_search":
                speak("Searching on Wikipedia...")
                query = query.replace('wikipedia', '')
                try:
                    results = wikipedia.summary(query, sentences=2)
                    speak("According to Wikipedia:")
                    speak(results)
                except wikipedia.DisambiguationError:
                    speak("The topic is ambiguous. Please be more specific.")
                except wikipedia.PageError:
                    speak("Sorry, I couldn't find any information on that topic.")
            elif intent == "take_picture":
                image_path = capture_image()
                if image_path:
                    detected_faces = detect_faces(image_path)
                    if detected_faces:
                        speak("I detected faces. See the captured image.")
                    else:
                        speak("No faces detected.")
            elif intent == "generative_ai":
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(query)
                out = response.text.replace('*', '')
                speak(out)

            elif intent == "exit":
                speak("Goodbye! Have a great day.")
                break
            else:
                speak("I'm not sure how to respond to that. Would you like to teach me?")
                correction = voice_input()
                if correction:
                    custom_responses[query] = correction
                    save_custom_responses()
                    speak("Got it! I'll remember this for next time.")

