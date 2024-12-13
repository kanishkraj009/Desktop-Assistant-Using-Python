# Desktop-Assistant-Using-Python


This project is a Python-based AI-powered virtual assistant that integrates speech recognition, text-to-speech, and advanced generative AI capabilities. It allows users to interact with the system through voice commands and get AI-generated responses, Wikipedia summaries, and more.

## Features
- **Speech Recognition**: Converts user voice input into text using the `SpeechRecognition` library.
- **Text-to-Speech**: Speaks responses aloud using the `pyttsx3` library.
- **AI Content Generation**: Generates responses using the `google.generativeai` Gemini API.
- **Wikipedia Integration**: Fetches summaries from Wikipedia for queries containing "Wikipedia."
- **Greeting Functionality**: Greets the user based on the current time of day.

## Technologies Used
- **Python Libraries**: 
  - `pyttsx3` for text-to-speech.
  - `speech_recognition` and `pyaudio` for voice input.
  - `wikipedia` for fetching web-based information.
  - `google.generativeai` for AI-generated content.
- **Flask** (Optional): Ready for web integration if needed.

## How to Use
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Configure your Google Generative AI API key.
4. Run the script and interact with the assistant through voice commands.

## Future Improvements
- Add support for more APIs for diverse functionalities.
- Enhance error handling and robustness.
- Provide a GUI for a better user experience.

Feel free to fork, contribute, and adapt the project to your needs!
