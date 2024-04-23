import streamlit as st
import make_dataset
from tensorflow.keras.models import load_model
import os
import numpy as np
from pydub import AudioSegment

# Constants
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]


def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]

# Function to run the app
def run_app(song_path):
    # Call the AppManager
    X = make_dataset.make_dataset_dl(song_path)
    model = load_model("custom_cnn_2d.h5")
    preds = model.predict(X)
    votes = majority_voting(preds, genres)
    print("{} is a {} song".format(song_path, votes[0][0]))
    print("most likely genres are: {}".format(votes[:3]))
    # Return the genre prediction
    return votes[0][0]

# Main function to run the app
def main():
    # Set the title and description
    st.title("Music Genre Recognition")
    st.write("Upload an audio file (.wav or .mp3) and we'll predict its genre.")

    # Get user input
    #type_of_app = st.sidebar.selectbox("Select type of app:", ["dl", "ml"])
    song_path = st.file_uploader("Upload a song file (.wav or .mp3 format only):", type=["wav","mp3"])

    os.makedirs("temp",exist_ok=True)

    # Check if user uploaded a file
    if song_path is not None:
        # Save the file
        file_path = os.path.join("temp", song_path.name)
        filename = file_path
        if filename.lower().endswith(".mp3"):
            input_path = filename
            output_path = os.path.splitext(filename)[0] + ".wav"

            # Load the audio file into PyDub
            audio = AudioSegment.from_mp3(input_path)

            # Export the audio to WAV
            audio.export(output_path, format="wav")

            # Remove the original MP3 file
            os.remove(input_path)
        with open(file_path, "wb") as f:
            f.write(song_path.getbuffer())

        # Run the app and get the output
        genre = run_app(file_path)
        st.audio(file_path, format='audio/mp3')

        # Display the output
        st.success("The predicted genre is: {}".format(genre))

# Run the app
if __name__ == '__main__':
    main()
