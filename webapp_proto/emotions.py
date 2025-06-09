import random

EMOTIONS = ["aggressive", "happy", "relaxed", "sad"]
EMOTIONS_TO_INDEX = {emotion: index for index, emotion in enumerate(EMOTIONS)}
EMOTIONS_TO_COLOR = {
    "aggressive": "#FF0000",  # Red
    "happy": "#FFFF00",       # Yellow
    "relaxed": "#00FF00",     # Green
    "sad": "#0000FF"          # Blue
}

def randomEmotion():
    """Return a random emotion from the list"""
    return random.choice(EMOTIONS)