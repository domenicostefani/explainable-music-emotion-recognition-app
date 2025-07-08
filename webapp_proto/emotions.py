import os, random
import io, base64 # For base64 encoding of images


EMOTIONS = ["aggressive", "happy", "relaxed", "sad"]
EMOTIONS_TO_INDEX = {emotion: index for index, emotion in enumerate(EMOTIONS)}
EMOTIONS_TO_COLOR = {
    "aggressive": "#FF8E8E",  
    "happy": "#e8cf2e", 
    "relaxed": "#B8DCC6", 
    "sad": "#6BC5E0",  
}

#"#FF0000",  # Red
#"#FFFF00",       # Yellow
#"#00FF00",     # Green
#"#0000FF"          # Blue

# def randomEmotion():
#     """Return a random emotion from the list"""
#     return random.choice(EMOTIONS)

def randomEmotion():

    # Generate random numbers (as many as in EMOTIONS)
    random_numbers = [random.random() for _ in EMOTIONS]
    # Apply softmax to the random numbers
    exp_values = [2.718 ** num for num in random_numbers]
    assert len(exp_values) == len(EMOTIONS)
    total = sum(exp_values)
    probabilities = [value / total for value in exp_values]

    # Ensure probabilities sum to 1
    assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities do not sum to 1"

    # return value with max probability, plus all probabilities
    max_index = probabilities.index(max(probabilities))
    emotion = EMOTIONS[max_index]
    return emotion, {e:m for e,m in zip(EMOTIONS,probabilities)}


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
        assert len(predictions) <= 0 or type(predictions[0]) == dict, "predictions must be a list of dictionaries"

        if len(predictions) == 0:
            return '', dict.fromkeys(self.class_names, 0.0)

        total_per_emotion = {emotion: 0 for emotion in predictions[0].keys()}
        for prediction in predictions:
            for emotion, value in prediction.items():
                total_per_emotion[emotion] += value

        final_prediction = {emotion: total / len(predictions) for emotion, total in total_per_emotion.items()}
        predicted_emotion = max(final_prediction, key=final_prediction.get)

        return predicted_emotion, final_prediction
    

    def getPredictFunction(self):
        """Returns directly the predict function of the model"""
        def predict_function(mel_segment):
            """Predicts the emotion for a batch of mel segments"""
            predictions = self.model.predict(mel_segment)
            return predictions
        return predict_function
    

from lime import lime_image

def get_heatmap(explanation_, instance_, fig_path_, segment_duration_s=3, mask=None, display= False):

    # m. select the class with highest score
    class_index = explanation_.top_labels[0]    
    # map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation_.local_exp[class_index])

    print('heatmap: '+'\n'.join([str(e) for e in dict_heatmap.items()]))

    heatmap = np.vectorize(dict_heatmap.get)(explanation_.segments)    

    print('shape of heatmap:', heatmap.shape)

    
    plt.figure(figsize=(8, 6)) 
    image_background = instance_[:,:,0]
    # plot. use symmetrical colorbar that makes more sense
    if mask is None:
        plt.imshow(image_background, aspect='auto', origin='lower', cmap='gray', alpha=1)    
        plt.imshow(heatmap, aspect='auto', cmap='RdBu_r', vmin=-heatmap.max(), vmax=heatmap.max(),  origin='lower', alpha=0.8)
        plt.colorbar()
    else:
        plt.imshow(image_background, aspect='auto', origin='lower', cmap='gray', alpha=0.3) 
        plt.imshow(mask, aspect='auto', origin='lower', cmap='Blues', alpha=0.9) 

    # plt.imshow(heatmap, aspect='auto', cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max(), alpha=0.8) #origin='lower' to have (0, 0) left bottom
    
    # print('xlim:', plt.xlim())
    ofst = 0.5
    plt.xlim(-ofst, heatmap.shape[1]-ofst)  # set xlim to match the spectrogram width
    plt.ylim(-ofst, heatmap.shape[0]-ofst)  # set ylim to match

    # xticks = np.arange(0, heatmap.shape[1]-0.5, segment_duration_s * 10)  # every segment_duration_s seconds
    interval = heatmap.shape[1]/segment_duration_s/2
    xticks = np.arange(-ofst, heatmap.shape[1]-ofst+interval, interval)  # every segment_duration_s seconds
    
    x_to_time = lambda x: x/heatmap.shape[1] * segment_duration_s  # convert index to time in seconds
    plt.xticks(xticks, ['%.2f'%(x_to_time(t+ofst)) for t in xticks], rotation=0)
    # plt.xticks(xticks, ['%.1f'%t for t in xticks], rotation=0)

    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    if fig_path_ != False: plt.savefig(fig_path_)
    if display:
        plt.show()

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    heatmap_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return heatmap_url, heatmap


def run_lime_explanation(time_data, waveform_data, emotionClassifier):
    mel_segment = get_input_sample(waveform_data)

    # list_mel_segments: list of samples

    top_labels = len(EMOTIONS)

    # see the prediction
    print(">>> LIME calls classifier with batch of shape:", np.shape(mel_segment))
    prediction = emotionClassifier.predict_segment(mel_segment)
    print(f'prediction', prediction)

    # # get explanation
    explainer = lime_image.LimeImageExplainer()
    instance = mel_segment[0]  # to avoid batch dimension (128, 130, 3)
    explanation = explainer.explain_instance(instance, emotionClassifier.getPredictFunction(), top_labels=top_labels, hide_color=0,
                                            num_samples=20, segmentation_fn=segmentation_meaningful_regions)
    # print('pred:', explanation.top_labels[0], 'true:', LABEL) # prediction

    # # get and plot heatmap
    LIME_heatmap_url, LIME_heatmap = get_heatmap(explanation, instance, fig_path_=False)
    # explanation.local_exp[explanation.top_labels[0]]



    # ---------------------- #
    #  For detailed heatmap  #
    # ---------------------- #

    def get_positive_heatmap(image, normalize=True):
        positive_image = image.copy()
        positive_image [ positive_image < 0 ] = 0
        if normalize:
            positive_image = (positive_image - positive_image.min()) / (positive_image.max() - positive_image.min())        
        return positive_image


    # get a dictionary to map each segment ID with the corresponding value
    def get_value_for_each_segment(seg_map, image):
        # get segmentation indeces
        unique_segments = np.unique(seg_map)
        
        dict_segment_values = {}    
        for segment in unique_segments:    
            mask = seg_map == segment  # create a mask for that segment ID
            pixel_values = image[mask]  # take all the pixel values in the region
            unique_pixel_val = np.unique(pixel_values)  # all pixels are the same so une unique
            dict_segment_values[segment] = unique_pixel_val[0]   
        return dict_segment_values



    # get segmentation map
    segmentation_map = segmentation_meaningful_regions(LIME_heatmap)

    # make LIME and SHAP heatmaps positive and normalized
    LIME_heatmap_pos = get_positive_heatmap(LIME_heatmap)
    #SHAP_heatmap_pos = get_positive_heatmap(SHAP_heatmap)
    #GRAD_heatmap_pos = get_positive_heatmap(GradCAM_heatmap_avg)

    dict_segment_val = get_value_for_each_segment(segmentation_map, LIME_heatmap_pos)

    #best_segments = sorted(dict_segment_val, key=dict_segment_val.get, reverse=True)
    best_regions = sorted(dict_segment_val, key=dict_segment_val.get, reverse=True)

    # ----------------------------------
    #plot best N regions
    N=1
    epsilon = 0.0001  # to avoid buchetti
    image = instance.copy() + epsilon
    # image_height, image_width = image.shape[:2]

    # make a mask for the best segments
    mask = np.isin(segmentation_map, best_regions[0:N])
    # make image of relevat parts
    highlighted_image = np.zeros_like(image) 
    highlighted_image[mask] = image[mask] 


    mask_rel = highlighted_image[:, :, 0] != 0
    detailed_Heatmap_url, _ = get_heatmap(explanation, instance, fig_path_=False, mask=np.ma.masked_where(~mask_rel, highlighted_image[:, :, 0]))

    res = {
        "heatmap_overall": LIME_heatmap_url,
        "heatmap_highlight": detailed_Heatmap_url,
    }

    return res