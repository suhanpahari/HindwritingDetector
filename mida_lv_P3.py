import speech_recognition as sr
import tkinter as tk

# Create a tkinter window
window = tk.Tk()

# Create a label to display the live caption
caption_label = tk.Label(window, text="Live Caption", font=("Arial", 16))
caption_label.pack()

def update_live_caption():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        # Transcribe the captured audio
        transcript = r.recognize_google(audio)
        # Update the caption label with the transcribed text
        caption_label.config(text=transcript)
    except sr.UnknownValueError:
        # If speech cannot be recognized, display an appropriate message
        caption_label.config(text="Unable to transcribe speech")
    except sr.RequestError as e:
        # Handle any errors that occur during speech recognition
        caption_label.config(text="Error: {0}".format(e))

    # Schedule the next update after a short delay (e.g., 100 milliseconds)
    window.after(100, update_live_caption)

# Start the live captioning
update_live_caption()

# Start the tkinter event loop
window.mainloop()
