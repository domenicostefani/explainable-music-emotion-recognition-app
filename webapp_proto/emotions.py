import os

EMOTIONS = ["aggressive", "happy", "relaxed", "sad"]
EMOTIONS_TO_INDEX = {emotion: index for index, emotion in enumerate(EMOTIONS)}
EMOTIONS_TO_COLOR = {
    "aggressive": "#FF0000",  # Red
    "happy": "#FFFF00",       # Yellow
    "relaxed": "#00FF00",     # Green
    "sad": "#0000FF"          # Blue
}

# def randomEmotion():
#     """Return a random emotion from the list"""
#     return random.choice(EMOTIONS)

# def randomEmotion():

#     # Generate random numbers (as many as in EMOTIONS)
#     random_numbers = [random.random() for _ in EMOTIONS]
#     # Apply softmax to the random numbers
#     exp_values = [2.718 ** num for num in random_numbers]
#     assert len(exp_values) == len(EMOTIONS)
#     total = sum(exp_values)
#     probabilities = [value / total for value in exp_values]

#     # Ensure probabilities sum to 1
#     assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities do not sum to 1"

#     # return value with max probability, plus all probabilities
#     max_index = probabilities.index(max(probabilities))
#     emotion = EMOTIONS[max_index]
#     return emotion, probabilities


from tensorflow.keras.models import load_model
import numpy as np
import librosa
import matplotlib.pyplot as plt


# functions
# get empirically the bin position of a specific frequency
def get_mel_coeff_from_frequency(freq, sr=22050):    
    duration = 2
    sampling_rate = sr    
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)  # time samples 
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)   # generate wave     
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=2048, hop_length=512, n_mels=128)  # get mel spec   
    slice = mel_spectrogram[:,30]   # get a slice of the spec    
    return np.argmax(slice)


#(image, n_time_splits=5, low=250, mid_low=1000, mid_high=3000)
def segmentation_meaningful_regions(image, n_time_splits=3, low=250, mid_low=500, mid_high=2000):
    
    low_bin = get_mel_coeff_from_frequency(low)       # 250 -> band 9
    mid_low_bin = get_mel_coeff_from_frequency(mid_low)   # 500 -> band 18
    mid_high_bin = get_mel_coeff_from_frequency(mid_high)  # 2000 -> band 64
    segments = np.zeros(image.shape[:2], dtype=int)   # avoid channel dimension
    
    # find lenghth of each rectangle
    rect_length = image.shape[1] // n_time_splits
    id = 0
    for i in range(n_time_splits):        
        x_start = i * rect_length        
        # assign segment number to the upper rect in this "time" region
        segments[: , x_start : ] = id
        segments[low_bin : , x_start : ] = id + 1
        segments[mid_low_bin :,  x_start : ] = id + 2
        segments[mid_high_bin:,  x_start : ] = id + 3
        id += 4
    return segments 


# si usa? altrimenti cancella
def get_significan_segments(explanation_, fig_path_, num_features_=5):

    COLOR = 'viridis'  # 'magma'
    temp, mask = explanation_.get_image_and_mask(explanation_.top_labels[0], positive_only=True, num_features=num_features_, hide_rest=True)  # set hide_rest=True to see entire image
       
    plt.figure(figsize=(8, 6))
    temp_norm = (temp - temp.min()) / (temp.max() - temp.min())  
    plt.imshow(mark_boundaries(temp_norm, mask), origin='lower', aspect='auto')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if fig_path_ != False: plt.savefig(fig_path_)    
    plt.show()


# get mel log mel spec and normalize
def get_input_sample(segment_, sampling_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
    mel = librosa.feature.melspectrogram(y=segment_, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_norm = ( mel_db - mel_db.min() ) / ( mel_db.max() - mel_db.min() )  # normalize 0-1

    # add rgb channels and expand dims
    mel_rgb = np.repeat(mel_db_norm[..., np.newaxis], 3, axis=-1)  # added 3 channels
    return np.expand_dims(mel_rgb, axis=0)

def split_song(audiosignal, segment_duration_s=3, sampling_rate=22050, overlap=0):
    
    y = audiosignal
    
    segment_samples = segment_duration_s * sampling_rate  # number of samples in each segment
    hop_length_segm = int(segment_samples * (1 - overlap))

    max_start = len(y) - segment_samples
    num_segments = max_start // hop_length_segm + 1

    mel_segments = []  # save preprocessed input sample for each segment
    audio_segments = []
    start_end = []
    for i in range(num_segments):
        start = i * hop_length_segm        
        end = start + segment_samples
        segment = y[start:end]  
        start_end.append((start, end))
        audio_segments.append(segment)                        

        # get spectrogram and prepar
        mel_rgb_batch = get_input_sample(segment)
        mel_segments.append(mel_rgb_batch)

    return mel_segments, audio_segments, start_end

class EmotionClassifier:
    def __init__(self, model_path):
        assert model_path is not None, "Model path cannot be None"
        assert model_path.endswith('.h5'), "Model path must point to a .h5"
        assert os.path.exists(model_path), "Model file does not exist at the specified path ("+model_path+"). We currently are in the directory: " + os.getcwd()
        print("", flush=True)
        self.model = load_model(model_path)
        self.class_names = EMOTIONS
        self.class_indices = EMOTIONS_TO_INDEX

    def predict_segment(self, mels):
        predictions = self.model.predict(mels)[0]
        print('self.model.predict returned shape:', predictions.shape, flush=True)
        predicted_index = np.argmax(predictions)
        predicted_emotion = self.class_names[predicted_index]
        predictions = [float(pred) for pred in predictions]  # convert to float for JSON serialization
        return predicted_emotion, predictions
    
    def majority_vote(self, predictions):
        assert type(predictions) == list, "predictions must be a list"
        assert type(predictions[0]) == dict, "predictions must be a list of dictionaries"

        total_per_emotion = {emotion: 0 for emotion in predictions[0].keys()}
        for prediction in predictions:
            for emotion, value in prediction.items():
                total_per_emotion[emotion] += value

        final_prediction = {emotion: total / len(predictions) for emotion, total in total_per_emotion.items()}
        predicted_emotion = max(final_prediction, key=final_prediction.get)

        return predicted_emotion, final_prediction