import random

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
    return emotion, probabilities