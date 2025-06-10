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
            
            sf.write(filename, audio_data, 48000, subtype='PCM_16')
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

    inbufnp*= 6.0
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
    client.connect(client.outports[0], jackAudioOut)






# Global variable to store the current waveform data
current_waveform = None
waveform_json = None
recording_status = {"is_recording": False, "duration": 0, "start_time": None}

def generate_sinusoid(duration):
    """Generate a sinusoid for the specified duration at 48kHz sample rate"""
    sample_rate = 48000  # Hz
    frequency = 20  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sinusoid with some modulation to make it more interesting
    waveform = np.sin(2 * np.pi * frequency * t) * np.exp(-t/20)  # Decaying sinusoid
    waveform += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # Add harmonic
    
    # Normalize to [-1, 1]
    if len(waveform) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    
    return t, waveform

def read_wav_file(filename):
    """Read a WAV file and return time and waveform data"""
    try:
        data, sample_rate = sf.read(filename, dtype='float32')
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

def create_waveform_plot(time_data, waveform_data, emotionLabels=None, title="Audio Waveform", highlight_slice=None):
    """Create a waveform plot and return as base64 encoded image"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_data, waveform_data, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    # plt.ylabel('Amplitude')
    # Remove yticks
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # maxw, minw = np.max(np.abs(waveform_data)), np.min(np.abs(waveform_data))
    # maxw = max(maxw, 1e-6)
    # minw = min(minw, -1e-6)
    # maxw = max(abs(maxw), abs(minw))
    # minw = -maxw
    minw, maxw = (-1.0, 1.0)  # Fixed range for better visualization

    # ax.set_ylim(-1.1, 1.1)
    ax.set_ylim(minw*1.1, maxw*1.1)

    x_min, x_max = ax.get_xlim()
    # print(f"x_min: {x_min}, x_max: {x_max}")
    trans = ax.transData.transform
    sliceBoundaries = []
    
    if emotionLabels is not None:
        # get pixel boundaries for each emotion label
        right_bound_pixel, _ = trans((x_max, 0))
        left_bound_pixel, _ = trans((x_min, 0))
        for i, label in enumerate(emotionLabels):
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
        if highlight_slice is None and len(time_data) > 0:
            duration = time_data[-1]
            slice_duration = 3  # seconds
            num_slices = len(emotionLabels)
            
            for i in range(1, num_slices):
                slice_time = i * slice_duration
                if slice_time < duration:
                    plt.axvline(x=slice_time, color='r', linestyle='--', alpha=0.6)
            
            # Add slice labels
            for i in range(num_slices):
                slice_start = i * slice_duration
                slice_center = slice_start + slice_duration / 2
                if slice_center < duration:
                    label = emotionLabels[i] if emotionLabels else ""
                    facecolor = emotions.EMOTIONS_TO_COLOR[label] if label in emotions.EMOTIONS_TO_COLOR else 'lightgrey'
                    plt.text(slice_center, 0.9, label, ha='center', va='center', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolor, alpha=0.7),
                            fontsize=14, fontweight='bold')
                
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url, sliceBoundaries

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

@app.route('/clear_waveform', methods=['POST'])
def clear_waveform():
    """Clear the current waveform data"""
    global current_waveform
    current_waveform = None

    return jsonify({"status": "ready"})

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

    

@app.route('/compute_waveform')
def compute_waveform():
    """Generate and return the waveform data"""
    global current_waveform, waveform_json, recFilename, recording_status

    print("Computing waveform...")
    
    if not recording_status["is_recording"] and recording_status["duration"] > 0:
        # Generate the sinusoid for the recorded duration
        duration = recording_status["duration"]
        _, waveform_data = generate_sinusoid(duration)
        while recFilename is None:
            print("Waiting for recording to finish...")
            time.sleep(0.1)

        print(f"Reading WAV file: {recFilename}")
        waveform_data, sr = read_wav_file(recFilename) if recFilename else ([], [])
        duration = len(waveform_data) / sr

        time_data = np.linspace(0, duration, len(waveform_data))

        # time_data, waveform_data = downscale_waveform(time_data, waveform_data, max_X_samples = 1000, max_Y_samples = 1000)
        duration = time_data[-1] if len(time_data) > 0 else 0

        # print(f"Duration of recorded audio: {duration} seconds")
        # print(f"Number of samples: {len(waveform_data)}")
        # print(f"Time Data: ", time_data[:10], "...")  # Print first 10 samples for debugging
        # print(f"Waveform Data: ", waveform_data[:10], "...")  # Print first 10 samples for debugging

        current_waveform = (time_data, waveform_data)
        
        # Calculate number of slices
        num_slices = int(math.floor(duration / 3))  # 3-second slices
        # Create the plot
        emotionLabels = [emotions.randomEmotion() for _ in range(num_slices)]
        plot_url, sliceBoundaries = create_waveform_plot(time_data, waveform_data, emotionLabels)
        

        colors = [emotions.EMOTIONS_TO_COLOR[label] for label in emotionLabels]

        slices = [{
                    "id": i, 
                    "label": emotionLabels[i],
                    "color": colors[i],
                    "left": sliceBoundaries[i][0],
                    "right": sliceBoundaries[i][1],
                    "width": sliceBoundaries[i][1] - sliceBoundaries[i][0],
                   } for i in range(num_slices)]
        
        waveform_json = jsonify({
            "status": "ready",
            "filename": recFilename,
            "plot": plot_url,
            "duration": duration,
            "slices": slices,
        })

        # Save to file
        json_filename = os.path.splitext(recFilename)[0] + ".json"
        with open(os.path.splitext(recFilename)[0]+".json", "w") as f:
            f.write(waveform_json.get_data(as_text=True))
        print(f"Waveform data computed and saved to {json_filename}")

        return waveform_json

    else:
        return jsonify({"status": "not_ready"})

@app.route('/slice/<int:slice_id>')
def view_slice(slice_id):
    """View a specific 3-second slice"""
    global current_waveform
    
    if current_waveform is None:
        return "No waveform data available", 404
    
    time_data, waveform_data = current_waveform
    slice_duration = 3  # seconds
    sample_rate = 48000
    
    # Calculate slice boundaries
    start_sample = slice_id * slice_duration * sample_rate
    end_sample = min((slice_id + 1) * slice_duration * sample_rate, len(time_data))
    
    # Extract slice data
    slice_time = time_data[start_sample:end_sample]
    slice_waveform = waveform_data[start_sample:end_sample]
    
    # Adjust time to start from 0 for the slice
    slice_time_adjusted = slice_time - slice_time[0]
    
    # Create zoomed plot
    slice_label = chr(ord('A') + slice_id)
    plot_url,_ = create_waveform_plot(
        slice_time_adjusted, 
        slice_waveform, 
        title=f"Audio Waveform - Slice {slice_label} (Zoomed)",
        highlight_slice=True
    )
    
    return render_template('slice.html', 
                         slice_label=slice_label,
                         slice_id=slice_id,
                         plot=plot_url)

# Create templates directory and HTML files
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))



if __name__ == '__main__':
    # Create templates before running the app
    print("Flask app starting...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True)