from flask import Flask, render_template, jsonify, request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import time
import threading
from datetime import datetime
import emotions, math, jack, soundfile as sf, librosa
from scipy.interpolate import interp1d
import sox

def sox_length(path):
    try:
        length = sox.file_info.duration(path)
        return length
    except:
        return None

CLASSIFIER_SR = 22050  # Sample rate for classifier
HIGHRES_SR = 48000  # Sample rate for high-resolution audio

class MonoAudioRecorder:
    def __init__(self):
        self.recording = False
        self.start_time = None
        self.duration = 0
        self.buffer = []

    def start(self):
        self.recording = True
        self.start_time = time.time()
        self.duration = 0

    def stop(self):
        if self.recording:
            self.duration = time.time() - self.start_time
            self.recording = False

    def process(self, buffer):
        """Process audio frames"""
        if self.recording:
            self.buffer.extend(buffer)

    def getBuffer(self):
        """Get the recorded audio buffer"""
        return np.array(self.buffer, dtype=np.float32)
    
    def saveToFile(self, filename):
        """Save the recorded audio to a file"""
        if self.buffer:
            audio_data = np.array(self.buffer, dtype=np.float32)
            
            sf.write(filename, audio_data, HIGHRES_SR, subtype='PCM_16')
            print(f"Audio saved to {filename}")
        else:
            print("No audio data to save.")
    
    def clearBuffer(self):
        """Clear the recorded audio buffer"""
        self.buffer = []

recorder = MonoAudioRecorder()
recFilename = None # To be locked and written from different threads


app = Flask(__name__)

# Close all jack clients
client = jack.Client('MusicEmotionXAI', session_id='MusicEmotionXAI')

event = threading.Event()

time.sleep(2)  # 
print('JACK server started!')
# create two port pairs
in1 = client.inports.register(f'input_{1}')
out1 = client.outports.register(f'output_{1}')

print('waiting for JACK server to start...')

@client.set_process_callback
def process(frames):
    assert len(client.inports) == len(client.outports)
    assert frames == client.blocksize
        
    # Double the volume of the input signal
    inbuf = memoryview(in1.get_buffer()).cast('f')


    inbufnp = np.frombuffer(inbuf, dtype=np.float32)

    # inbufnp*= 6.0 # Amplify the input signal by 6x
    recorder.process(inbufnp)  # Process the input buffer

    outbuf = memoryview(out1.get_buffer()).cast('f')

    
    outbufnp = np.frombuffer(outbuf, dtype=np.float32)
    if len(inbufnp) > 0:
        if recorder.recording:
            outbufnp[:] = inbufnp
        else:
            # If not recording, just pass the input to output
            outbufnp[:] = [0.0] * len(inbufnp)  # Mute output if not recording



@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()


client.activate()


capture = client.get_ports(is_physical=True, is_output=True)
if not capture:
    raise RuntimeError('No physical capture ports')

for jackAudioIn in capture:
    client.connect(jackAudioIn, client.inports[0])

playback = client.get_ports(is_physical=True, is_input=True)
if not playback:
    raise RuntimeError('No physical playback ports')

for jackAudioOut in playback:
    if not "midi" in jackAudioOut.name.lower():
        client.connect(client.outports[0], jackAudioOut)






# Global variable to store the current waveform data
current_waveform = None
waveform_json = None
recording_status = {"is_recording": False, "duration": 0, "start_time": None}

# def generate_sinusoid(duration, sample_rate=None):
#     """Generate a sinusoid for the specified duration at 48kHz sample rate"""
#     sample_rate = HIGHRES_SR  # Hz
#     frequency = 20  # Hz (A4 note)
    
#     # Generate time array
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
    
#     # Generate sinusoid with some modulation to make it more interesting
#     waveform = np.sin(2 * np.pi * frequency * t) * np.exp(-t/20)  # Decaying sinusoid
#     waveform += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # Add harmonic
    
#     # Normalize to [-1, 1]
#     if len(waveform) > 0:
#         waveform = waveform / np.max(np.abs(waveform))
    
#     return t, waveform

def read_wav_file(filename, sample_rate=None):
    """Read a WAV file and return time and waveform data"""
    try:
        if sample_rate is None:
            # data, sample_rate = sf.read(filename, dtype='float32')
            data, sample_rate = librosa.load(filename, sr=None, mono=True, dtype='float32')
        else :
            # data, sample_rate = sf.read(filename, dtype='float32', samplerate=sample_rate)
            data, sample_rate = librosa.load(filename, sr=sample_rate, mono=True, dtype='float32')

        if len(data.shape) > 1:  # If stereo, take only one channel
            data = data[:, 0]
        return data, sample_rate
    except Exception as e:
        print(f"Error reading WAV file {filename}: {e}")
        return None, None
    
# def downscale_waveform(time_data, waveform_data, max_X_samples = 1000, max_Y_samples = 1000):
#     """Downscale waveform data to fit within specified sample limits"""
#     if len(time_data) > max_X_samples:
#         # Downsample time data
#         indices = np.linspace(0, len(time_data) - 1, max_X_samples).astype(int)
#         time_data = time_data[indices]
#         waveform_data = waveform_data[indices]

#     if len(waveform_data) > max_Y_samples:
#         # Downsample waveform data
#         indices = np.linspace(0, len(waveform_data) - 1, max_Y_samples).astype(int)
#         waveform_data = waveform_data[indices]

#     return time_data, waveform_data

# def create_waveform_plot(time_data, waveform_data, emotionLabels=None, emotionProbabilities=None, title="Audio Waveform", highlight_slice=None,figure_size=(12, 6), display=True, sliceDuration=3, sliceStart=0.0):
#     """Create a waveform plot and return as base64 encoded image"""
#     fig, ax = plt.subplots(figsize=figure_size)
#     ax.plot(time_data, waveform_data, 'b-', linewidth=0.5)
#     ax.set_xlabel('Time (seconds)')
#     if highlight_slice is not None:
#         plt.ylabel('Amplitude')
#     else:
#         # Remove yticks
#         ax.set_yticks([])  # Hide y-axis ticks
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)
#     # maxw, minw = np.max(np.abs(waveform_data)), np.min(np.abs(waveform_data))
#     # maxw = max(maxw, 1e-6)
#     # minw = min(minw, -1e-6)
#     # maxw = max(abs(maxw), abs(minw))
#     # minw = -maxw
#     minw, maxw = (-1.0, 1.0)  # Fixed range for better visualization

#     # ax.set_ylim(-1.1, 1.1)
#     ax.set_ylim(minw*1.1, maxw*1.1)

#     x_min, x_max = ax.get_xlim()
#     # print(f"x_min: {x_min}, x_max: {x_max}")
#     trans = ax.transData.transform
#     sliceBoundaries = []
    
#     if emotionLabels is not None:
#         assert emotionProbabilities is not None, "Emotion probabilities must be provided if emotion labels are given"
#         assert len(emotionLabels) == len(emotionProbabilities), "Emotion labels and probabilities must have the same length"
#         # get pixel boundaries for each emotion label
#         right_bound_pixel, _ = trans((x_max, 0))
#         left_bound_pixel, _ = trans((x_min, 0))
#         for i, (label, probabilities) in enumerate(zip(emotionLabels, emotionProbabilities)):
#             assert len(probabilities) == len(emotions.EMOTIONS), "Probabilities must match the number of emotions"
#             time_left = i * 3  # 3 seconds per slice
#             time_right = (i + 1) * 3
#             slice_left, _ = trans((time_left, 0))
#             slice_right, _ = trans((time_right, 0))

#             def floatMap (x, in_min, in_max, out_min, out_max):
#                 """Map a float from one range to another"""
#                 return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#             relative_left_px = floatMap(slice_left, left_bound_pixel, right_bound_pixel, 0, 1)
#             relative_right_px = floatMap(slice_right, left_bound_pixel, right_bound_pixel, 0, 1)
#             sliceBoundaries.append((relative_left_px, relative_right_px))

#         # Add slice boundaries and labels for full waveform
#         if highlight_slice is None and len(time_data) > 0:
#             duration = time_data[-1]
#             num_slices = len(emotionLabels)

#             print(f"Duration: {duration}, Slice Duration: {sliceDuration}, Number of Slices: {num_slices}")
            
#             for i in range(0, num_slices+1):
#                 slice_time = i * sliceDuration+sliceStart
#                 if slice_time < duration:
#                     ax.axvline(x=slice_time, color='r', linestyle='--', alpha=0.6)
            
#             # Add slice labels
#             for i in range(num_slices):
#                 slice_start = i * sliceDuration
#                 slice_center = slice_start + sliceDuration / 2
#                 if slice_center < duration:
#                     label = emotionLabels[i] if i < len(emotionLabels) else "Unknown"
#                     textlabel = emotionLabels[i]+" %.1f%%"%(emotionProbabilities[i][emotionLabels[i]]*100) if emotionLabels else ""
#                     print('emotions.EMOTIONS_TO_COLOR',emotions.EMOTIONS_TO_COLOR)
#                     print('label',label)
#                     print('label in emotions.EMOTIONS_TO_COLOR.keys():', label in emotions.EMOTIONS_TO_COLOR.keys())
#                     facecolor = emotions.EMOTIONS_TO_COLOR[label] if label in emotions.EMOTIONS_TO_COLOR.keys() else 'lightgrey'
#                     plt.text(slice_center, 0.9, textlabel, ha='center', va='center', 
#                             bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolor, alpha=0.7),
#                             fontsize=14, fontweight='bold')
                
#     # // xticks every sliceDuration/2, starting from sliceStart
#     x_ticks = np.arange(sliceStart, time_data[-1], sliceDuration/2.0)
#     ax.set_xticks(x_ticks)
    
#     plt.tight_layout()
    
#     if display:
#         plt.show()
#     # Convert plot to base64 string
#     img = io.BytesIO()
#     plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode()
#     plt.close()
    
#     return plot_url, sliceBoundaries

def create_waveform_plot(time_data, waveform_data, emotionLabels=None, emotionProbabilities=None, title=None, highlight_slice=None,figure_size=(12, 6), display=True, sliceDuration=3, sliceStart=0, simple=False):
    """Create a waveform plot and return as base64 encoded image"""
    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(time_data, waveform_data, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    if highlight_slice is not None:
        plt.ylabel('Amplitude')
    else:
        # Remove yticks
        ax.set_yticks([])  # Hide y-axis ticks

    printVerbose = lambda *args, **kwargs: None  # Placeholder for verbose output

    if title:
        ax.set_title(title)
    if not simple:
        ax.grid(True, alpha=0.3)
    # maxw, minw = np.max(np.abs(waveform_data)), np.min(np.abs(waveform_data))
    # maxw = max(maxw, 1e-6)
    # minw = min(minw, -1e-6)
    # maxw = max(abs(maxw), abs(minw))
    # minw = -maxw
    minw, maxw = (-1.0, 1.0)  # Fixed range for better visualization

    # ax.set_ylim(-1.1, 1.1)
    if not simple:
        ax.set_ylim(minw*1.1, maxw*1.1)

    x_min, x_max = ax.get_xlim()
    # print(f"x_min: {x_min}, x_max: {x_max}")
    trans = ax.transData.transform
    sliceBoundaries = []

    # // xticks every sliceDuration/2, starting from sliceStart
    x_ticks = np.arange(max(sliceStart, time_data[0]), time_data[-1]*1.1, sliceDuration/2.0)
    ax.set_xticks(x_ticks)
    
    if emotionLabels is not None:
        assert emotionProbabilities is not None, "Emotion probabilities must be provided if emotion labels are given"
        assert len(emotionLabels) == len(emotionProbabilities), "Emotion labels and probabilities must have the same length"
        # get pixel boundaries for each emotion label
        right_bound_pixel, _ = trans((x_max, 0))
        left_bound_pixel, _ = trans((x_min, 0))
        for i, (label, probabilities) in enumerate(zip(emotionLabels, emotionProbabilities)):
            assert len(probabilities) == len(emotions.EMOTIONS), "Probabilities must match the number of emotions"
            time_left = i * 3  # 3 seconds per slice
            time_right = (i + 1) * 3
            slice_left, _ = trans((time_left, 0))
            slice_right, _ = trans((time_right, 0))

            def floatMap (x, in_min, in_max, out_min, out_max):
                """Map a float from one range to another"""
                return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

            relative_left_px = floatMap(slice_left, left_bound_pixel, right_bound_pixel, 0, 1)
            relative_right_px = floatMap(slice_right, left_bound_pixel, right_bound_pixel, 0, 1)
            sliceBoundaries.append((relative_left_px, relative_right_px))

        # Add slice boundaries and labels for full waveform
        if highlight_slice is None and len(time_data) > 0 and not simple:
            duration = time_data[-1]
            num_slices = len(emotionLabels)

            printVerbose(f"Duration: {duration}, Slice Duration: {sliceDuration}, Number of Slices: {num_slices}")
            
            for i in range(0, num_slices+1):
                slice_time = i * sliceDuration + sliceStart
                if slice_time < duration:
                    ax.axvline(x=slice_time, color='r', linestyle='--', alpha=0.6)
            
            # Add slice labels
            for i in range(num_slices):
                slice_start = i * sliceDuration
                slice_center = slice_start + sliceDuration / 2
                if slice_center < duration:
                    label = emotionLabels[i] if i < len(emotionLabels) else "Unknown"
                    printVerbose('emotionProbabilities[i]', emotionProbabilities[i])
                    textlabel = emotionLabels[i]+" %.1f%%"%(emotionProbabilities[i][emotionLabels[i]]*100) if emotionLabels else ""
                    printVerbose('emotions.EMOTIONS_TO_COLOR',emotions.EMOTIONS_TO_COLOR)
                    printVerbose('label',label)
                    printVerbose('label in emotions.EMOTIONS_TO_COLOR.keys():', label in emotions.EMOTIONS_TO_COLOR.keys())
                    facecolor = emotions.EMOTIONS_TO_COLOR[label] if label in emotions.EMOTIONS_TO_COLOR.keys() else 'lightgrey'
                    plt.text(slice_center, 0.9, textlabel, ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolor, alpha=0.7),
                            fontsize=14, fontweight='bold')
                
    
    
    plt.tight_layout()
    
    if display:
        plt.show()
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url, sliceBoundaries

def plot_emotion_graph(emotionProbabilities, startstop, audioduration, figure_size=(12, 6), display=False, smooth=True):
    sample2Seconds = lambda x: x / CLASSIFIER_SR
    startstop_s = [(sample2Seconds(start), sample2Seconds(stop)) for start, stop in startstop]

    fig, ax = plt.subplots(figsize=figure_size)

    # points should be a dict with a key per each emotion, and the value is a list of tuples (time, probability)
    points = {}
    for d, (start, stop) in zip(emotionProbabilities, startstop_s):
        for emotion, prob in d.items():
            if emotion not in points:
                points[emotion] = []
            # Calculate the time for this segment (center of segment)
            segment_time = (start + stop) / 2
            points[emotion].append((segment_time, prob))

    # Get the overall time range
    if startstop_s:
        overall_start = min(start for start, _ in startstop_s)
        overall_end = max(stop for _, stop in startstop_s)
    else:
        overall_start, overall_end = 0, 1

    # Create lines for each emotion (smooth or simple)
    for emotion, pts in points.items():
        if len(pts) == 0:
            continue
            
        # Sort points by time
        pts = sorted(pts, key=lambda x: x[0])
        point_times, point_probs = zip(*pts)
        
        emotion_capitalized = emotion.capitalize()
        color = emotions.EMOTIONS_TO_COLOR.get(emotion, 'lightgrey')
        
        if smooth and len(pts) >= 2:
            # Smooth interpolated lines
            # Extend the line to cover the full time range
            # Add points at the beginning and end with the same probability as the first/last points
            extended_times = [overall_start] + list(point_times) + [overall_end]
            extended_probs = [point_probs[0]] + list(point_probs) + [point_probs[-1]]
            
            # Create interpolation function
            # Use 'cubic' for smooth curves, 'linear' for straight lines between points
            try:
                interp_func = interp1d(extended_times, extended_probs, kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
            except:
                # Fall back to linear interpolation if cubic fails
                interp_func = interp1d(extended_times, extended_probs, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
            
            # Generate smooth time series for plotting
            smooth_times = np.linspace(overall_start, overall_end, 200)
            smooth_probs = interp_func(smooth_times)
            
            # Clip probabilities to [0, 1] range
            smooth_probs = np.clip(smooth_probs, 0, 1)
            
            # Plot the smooth line
            ax.plot(smooth_times, smooth_probs, 
                   label=emotion_capitalized, 
                   color=color, 
                   linewidth=2, alpha=0.8)
            
            # Plot the original data points
            ax.scatter(point_times, point_probs, 
                      alpha=0.7, 
                      color=color,
                      s=30, zorder=5)
        else:
            # Simple lines connecting points directly
            # Add start and end points to extend the line
            start_time, start_prob = pts[0]
            end_time, end_prob = pts[-1]
            extended_times = [overall_start] + list(point_times) + [overall_end]
            extended_probs = [start_prob] + list(point_probs) + [end_prob]
            
            # Plot simple line
            ax.plot(extended_times, extended_probs, 
                   linestyle='--', 
                   label=emotion_capitalized, 
                   color=color, 
                   linewidth=1, alpha=0.8)
            
            # Plot the original data points
            ax.scatter(point_times, point_probs, 
                      alpha=0.5, 
                      color=color)

    # Set x-axis ticks
    overall_end = max(overall_end, audioduration)  # Ensure end covers audio duration
    ax.set_xlim(overall_start, overall_end)

    xticks = np.arange(overall_start, overall_end + 1, 3)
    ax.set_xticks(xticks, [f"{t:.0f}" for t in xticks], rotation=0)

    # Add vertical dashed lines for each segment
    for start, end in startstop_s:
        ax.axvline(x=start, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=end, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Emotion Probability')
    # ax.set_ylim(0, 1)  # Ensure y-axis covers full probability range
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if display:
        plt.show()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url

def create_spectrogram_plot(time_data, waveform_data, sample_rate, title="Spectrogram", figure_size=(12, 6)):
    """Create a Spectrogram plot and return as base64 encoded image"""
    fig, ax = plt.subplots(figsize=figure_size)
    # Compute a MEL spectrogram with librosa
    S = librosa.feature.melspectrogram(y=waveform_data, sr=sample_rate, n_mels=128, fmax=20000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    # plt.colorbar(img, ax=ax, format='%+2.0f dB')

    print(f"Time data : {time_data[:3]}...{time_data[-4:]}")  # Print first 10 time data points for debugging
    
    zerobased_time = time_data - time_data[0]  # Adjust time to start from 0



    XAX_FACTOR = 6.0
    ax.set_xticks(zerobased_time[::int(len(zerobased_time)/XAX_FACTOR)])  # Show 10 ticks
    ax.set_xticklabels([f"{t:.1f}" for t in time_data[::int(len(time_data)/XAX_FACTOR)]], rotation=45)

    print(f"zerobased_time : {zerobased_time[:3]}...{zerobased_time[-4:]}")  # Print first 10 time data points for debugging
    print(f"len(zerobased_time): {len(zerobased_time)}")  # Print length of time data for debugging
    print(f"zerobased_time cut = {zerobased_time[::int(len(zerobased_time)/XAX_FACTOR)]}")  # Print first 10 ticks for debugging

    plt.tight_layout()

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    print(f"Plot url is: {plot_url[:50]}...")  # Print first 50 characters for debugging

    return plot_url


def track_recording_time():
    """Track recording time"""
    global recording_status, recorder, recFilename

    if recorder.recording != True:
        recorder.start()
    
    while recording_status["is_recording"]:
        if recording_status["start_time"]:
            current_time = time.time()
            recording_status["duration"] = current_time - recording_status["start_time"]
        time.sleep(0.1)
    
    recorder.stop()
    timedatestr = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/recording_{timedatestr}.wav"
    recorder.saveToFile(filename)
    recorder.clearBuffer()  # Clear buffer after saving

    # Lock recFilename and write the filename
    recFilename = filename

@app.route('/')
def index():
    # Check if we have a waveform to display
    show_waveform = current_waveform is not None and not recording_status["is_recording"]
    return render_template('index.html', show_waveform=show_waveform)


@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start the recording process"""
    global recording_status, current_waveform, waveform_json, recFilename

    recFilename = None  # Reset filename for new recording
    waveform_json = None  # Reset waveform JSON data
    current_waveform = None  # Reset current waveform data

    # Clear previous waveform data when starting new recording
    current_waveform = None
    
    recording_status = {
        "is_recording": True, 
        "duration": 0, 
        "start_time": time.time()
    }
    
    # Start recording time tracking in background
    recording_thread = threading.Thread(target=track_recording_time)
    recording_thread.daemon = True
    recording_thread.start()
    
    return jsonify({"status": "recording_started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop the recording process"""
    global recording_status
    

    if recording_status["is_recording"]:
        recording_status["is_recording"] = False
        final_duration = recording_status["duration"]
        return jsonify({"status": "recording_stopped", "duration": final_duration})
    else:
        return jsonify({"status": "not_recording"})
    


@app.route('/recording_status')
def get_recording_status():
    """Get current recording status"""
    return jsonify(recording_status)

@app.route('/get_waveform')
def get_waveform():
    """Get the current waveform data"""
    global waveform_json

    if waveform_json is not None:
        print("get_waveform(): Returning existing waveform data")
        return waveform_json if waveform_json is not None else jsonify({"status": "not_ready"})
    else:
        print("get_waveform(): Computing NEW waveform data...")
        return compute_waveform()

   
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change to the directory where this script is located


emotionClassifier = emotions.EmotionClassifier(model_path='../classifier/best_model_1_acc_53.14.h5') # Relative to the previous chdir


# Add these modifications to your existing Flask app

import threading
import json

# Global variables for slice management
slice_waveforms = {}  # Store precomputed slice data
slice_computation_status = {"computing": False, "completed": False}

def precompute_slice_waveforms(time_data, waveform_data, emotion_labels, filename_base, emotion_probabilities):
    """Precompute all slice waveforms in a background thread"""
    global slice_waveforms, slice_computation_status
    
    slice_computation_status["computing"] = True
    slice_computation_status["completed"] = False
    slice_waveforms = {}  # Clear existing data
    
    try:
        slice_duration = 3  # seconds
        sample_rate = HIGHRES_SR  # Assuming 48kHz sample rate
        duration = time_data[-1] if len(time_data) > 0 else 0
        num_slices = int(math.floor(duration / slice_duration))
        
        print(f"Precomputing {num_slices} slice waveforms and spectrograms ...")
        
        for slice_id in range(num_slices):
            print(f"Computing slice {slice_id + 1}/{num_slices}")
            
            # Calculate slice boundaries (same logic as original view_slice)
            start_sample = int(slice_id * slice_duration * sample_rate)
            end_sample = int(min((slice_id + 1) * slice_duration * sample_rate, len(time_data)))
            
            # Ensure we don't go out of bounds
            start_sample = min(start_sample, len(time_data) - 1)
            end_sample = min(end_sample, len(time_data))
            
            if start_sample >= end_sample:
                continue
                
            # Extract slice data
            slice_time = time_data[start_sample:end_sample+1]
            slice_waveform = waveform_data[start_sample:end_sample+1]
            
            if len(slice_time) == 0:
                continue
                
            
            # Create zoomed plot
            slice_label = emotion_labels[slice_id] if slice_id < len(emotion_labels) else f"Slice {slice_id}"
            slice_emotion_probabilities = emotion_probabilities[slice_id] if slice_id < len(emotion_probabilities) else None
            plot_url, _ = create_waveform_plot(
                slice_time, 
                slice_waveform, 
                title=f"Audio Waveform - {slice_label} (Zoomed)",
                highlight_slice=True,
                figure_size=(7, 6)
            )

            spec_url = create_spectrogram_plot(
                slice_time,
                slice_waveform, 
                HIGHRES_SR,  # Assuming 48kHz sample rate
                title=f"Spectrogram - {slice_label}",
                figure_size=(7, 6)
            )
            
            # Store the precomputed data
            slice_waveforms[slice_id] = {
                "slice_id": slice_id,
                "slice_label": slice_label,
                "plot_url": plot_url,
                "spec_url": spec_url,
                "time_data": slice_time.tolist(),
                "waveform_data": slice_waveform.tolist(),
                "duration": slice_time[-1] if len(slice_time) > 0 else 0,
                "emotion_probabilities": slice_emotion_probabilities
            }
        
        # Save precomputed slices to file
        slices_filename = f"{filename_base}_slices.json"
        with open(slices_filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for slice_id, data in slice_waveforms.items():
                serializable_data[str(slice_id)] = {
                    "slice_id": data["slice_id"],
                    "slice_label": data["slice_label"],
                    "plot_url": data["plot_url"],
                    "duration": data["duration"],
                    "emotion_probabilities": data["emotion_probabilities"],
                    # Note: not saving time_data and waveform_data to keep file size manageable
                }
            json.dump(serializable_data, f, indent=2)
        
        print(f"Slice waveforms precomputed and saved to {slices_filename}")
        
    except Exception as e:
        print(f"Error precomputing slice waveforms: {e}")
    finally:
        slice_computation_status["computing"] = False
        slice_computation_status["completed"] = True

from flask import send_file

@app.route('/recordings/<filename>')
def serve_audio(filename):
    """Serve audio files from the recordings directory"""
    # Make sure to validate the filename for security
    if not filename.endswith('.wav'):
        return "Invalid file type", 400
    
    # Construct the full path (adjust this to your actual recording directory)
    audio_path = os.path.join('recordings', filename)
    
    if not os.path.exists(audio_path):
        return "File not found", 404
    
    return send_file(audio_path, mimetype='audio/wav')





from werkzeug.utils import secure_filename

# Configure upload settings
UPLOAD_FOLDER = 'recordings'  # Create this directory
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

# Make sure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload and save audio file"""
    global recFilename, recording_status
    
    print(f"Received upload request")
    
    # Check if file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['audio_file']
    
    # Check if file was actually selected
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename and save
        filename = secure_filename(file.filename)
        
        # Add timestamp to avoid conflicts
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Store the full path for later use
        recFilename = filepath
        # recording_status["duration"] = 1  # You might want to get actual duration
        recording_status["duration"] = sox_length(filepath)
        
        print(f"File saved as: {filepath}")
        
        return jsonify({
            "status": "success", 
            "filename": unique_filename,
            "original_name": file.filename,
            "path": filepath
        })
    else:
        return jsonify({"status": "error", "message": "Invalid file type"}), 400

# Modify the compute_waveform function
@app.route('/compute_waveform')
def compute_waveform():
    """Generate and return the waveform data"""
    global current_waveform, waveform_json, recFilename, recording_status

    print("\n\nComputing waveform...")

    if not recording_status["is_recording"] and recording_status["duration"] >= 3:
        print("Recording has stopped, processing waveform...")
        # Generate the sinusoid for the recorded duration
        # duration = recording_status["duration"]
        # _, waveform_data = generate_sinusoid(duration)
        while recFilename is None:
            print("Waiting for recording to finish...")
            time.sleep(0.1)

        print(f"Reading WAV file: {recFilename}")
        waveform_data, sr = read_wav_file(recFilename, sample_rate=HIGHRES_SR) if recFilename else ([], [])
        assert waveform_data is not None, "Waveform data must not be None"
        assert len(waveform_data) > 0, "Waveform data must not be empty"
        duration = len(waveform_data) / sr

        time_data = np.linspace(0, duration, len(waveform_data))
        duration = time_data[-1] if len(time_data) > 0 else 0

        current_waveform = (time_data, waveform_data)
        
        # Calculate number of slices
        num_slices = int(math.floor(duration / 3))  # 3-second slices
        # Create the plot
        # random_emotions = [emotions.randomEmotion() for _ in range(num_slices)]

        lores_waveform_data, _ = read_wav_file(recFilename, CLASSIFIER_SR) if recFilename else ([], [])
        assert lores_waveform_data is not None, "Low-resolution waveform data must not be None"
        assert len(lores_waveform_data) > 0, "Low-resolution waveform data must not be empty"

        mel_segments, audio_segments, segment_startstop_smp = emotions.split_song(lores_waveform_data,
                                                              sampling_rate=CLASSIFIER_SR,
                                                              segment_duration_s=3,
                                                              overlap=0)
        pred_segment_emotions = []
        for mel_rgb_batch in mel_segments:
            # Predict emotions using the classifier
            pred_emotion_label, pred_probabilities = emotionClassifier.predict_segment(mel_rgb_batch)
            # print('shape of pred_emotion_label:', np.shape(pred_emotion_label))
            # print('shape of pred_probabilities:', np.shape(pred_probabilities))
            print("", flush=True)

            pred_emotions_dict = {emotion: prob for emotion, prob in zip(emotions.EMOTIONS, pred_probabilities)}

            pred_segment_emotions.append((pred_emotion_label, pred_emotions_dict))  # Store both emotion and probabilities
        
        overall_emotion, overall_probabilities = emotionClassifier.majority_vote([e[1] for e in pred_segment_emotions])
        print(f"Overall emotion: {overall_emotion}, Probabilities: {overall_probabilities}")


        # Now random emotions contain tuples of (emotion, probabilities)
        emotionLabels = [emotion[0] for emotion in pred_segment_emotions]
        emotionProbabilities = [emotion[1] for emotion in pred_segment_emotions]
        # print(f"Emotion labels: {emotionLabels}")
        # print(f"Emotion probabilities: {emotionProbabilities}")

        plot_url, sliceBoundaries = create_waveform_plot(time_data, waveform_data, emotionLabels, emotionProbabilities)

        emotion_chart_url = plot_emotion_graph(emotionProbabilities,segment_startstop_smp, duration)
        
        colors = [emotions.EMOTIONS_TO_COLOR[label] for label in emotionLabels]

        slices = [{
                    "id": i, 
                    "label": emotionLabels[i],
                    "probabilities": emotionProbabilities[i],
                    "color": colors[i],
                    "left": sliceBoundaries[i][0],
                    "right": sliceBoundaries[i][1],
                    "width": sliceBoundaries[i][1] - sliceBoundaries[i][0],
                   } for i in range(num_slices)]
        
        # Return an audio_url that can be played in an audio player in html
        audio_url = f"/recordings/{os.path.basename(recFilename)}"

        # Now Spectrogram with same fig_size 
        # spectrogram_url = create_spectrogram_plot(waveform_data, sr, title="Spectrogram of Recorded Audio", figure_size=(12, 6))
        
        waveform_json = jsonify({
            "status": "ready",
            "filename": os.path.basename(recFilename),
            "plot": plot_url,
            "emotion_chart": emotion_chart_url,
            "duration": duration,
            "slices": slices,
            "audio_url": audio_url,
            "overall_emotion": overall_emotion,
            "overall_probabilities": overall_probabilities,
        })

        # Save to file
        json_filename = os.path.splitext(recFilename)[0] + ".json"
        with open(json_filename, "w") as f:
            f.write(waveform_json.get_data(as_text=True))
        print(f"Waveform data computed and saved to {json_filename}")

        # Start precomputing slice waveforms in background
        filename_base = os.path.splitext(recFilename)[0]
        slice_thread = threading.Thread(
            target=precompute_slice_waveforms, 
            args=(time_data, waveform_data, emotionLabels, filename_base,emotionProbabilities)
        )
        slice_thread.daemon = True
        slice_thread.start()
        print("Started background slice precomputation")

        return waveform_json
    elif not recording_status["is_recording"] and recording_status["duration"] < 3:
        print("Recording too short, must be at least 3 seconds")
        return jsonify({"status": "not_ready", "message": "Recording too short, must be at least 3 seconds, found: %.2f seconds instead" % recording_status["duration"]})
    else:
        print("Waveform computation skipped, recording is still in progress or no recording data available")
        print(f"recording_status[is_recording']: {recording_status['is_recording']}")
        return jsonify({"status": "not_ready"})

# Add new route to check slice computation status
@app.route('/slice_status')
def get_slice_status():
    """Get slice computation status"""
    return jsonify(slice_computation_status)

# Modify the existing slice route to use precomputed data
@app.route('/slice/<int:slice_id>')
def view_slice(slice_id):
    """View a specific 3-second slice using precomputed data"""
    global slice_waveforms, current_waveform
    
    # Check if we have precomputed data
    if slice_id in slice_waveforms:
        print(f"Using precomputed data for slice {slice_id}")
        slice_data = slice_waveforms[slice_id]
        return render_template('slice.html', 
                             slice_label=slice_data["slice_label"],
                             slice_id=slice_id,
                             plot=slice_data["plot_url"],
                             spectrogram=slice_data["spec_url"],
                             audio_url=f"/recordings/{os.path.basename(recFilename)}",
                             slice_start=round(slice_data["time_data"][0],2),
                             slice_end=round(slice_data["time_data"][-1],2),
                             emotion_probabilities=slice_data["emotion_probabilities"],
        )
    
    # Fallback to original computation if precomputed data not available
    print(f"Precomputed data not available for slice {slice_id}, computing on-demand")
    
    if current_waveform is None:
        return "No waveform data available", 404
    
    time_data, waveform_data = current_waveform
    slice_duration = 3  # seconds
    sample_rate = HIGHRES_SR
    
    # Calculate slice boundaries
    start_sample = slice_id * slice_duration * sample_rate
    end_sample = min((slice_id + 1) * slice_duration * sample_rate, len(time_data))
    
    # Extract slice data
    slice_time = time_data[start_sample:end_sample]
    slice_waveform = waveform_data[start_sample:end_sample]

    
    # Create zoomed plot
    slice_label = chr(ord('A') + slice_id)
    plot_url, _ = create_waveform_plot(
        slice_time, 
        slice_waveform, 
        title=f"Audio Waveform - Slice {slice_label} (Zoomed)",
        highlight_slice=True
    )
    
    return render_template('slice.html', 
                         slice_label=slice_label,
                         slice_id=slice_id,
                         plot=plot_url)

# Add route to get precomputed slice data (for AJAX requests)
@app.route('/api/slice/<int:slice_id>')
def get_slice_data(slice_id):
    """Get precomputed slice data as JSON"""
    if slice_id in slice_waveforms:
        return jsonify(slice_waveforms[slice_id])
    else:
        return jsonify({"error": "Slice data not available"}), 404

# Modify the clear_waveform route to also clear slice data
@app.route('/clear_waveform', methods=['POST'])
def clear_waveform():
    """Clear the current waveform data and slice data"""
    global current_waveform, slice_waveforms, slice_computation_status
    
    current_waveform = None
    slice_waveforms = {}
    slice_computation_status = {"computing": False, "completed": False}

    return jsonify({"status": "ready"})

if __name__ == '__main__':
    # Create templates before running the app
    print("Flask app starting...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True)

