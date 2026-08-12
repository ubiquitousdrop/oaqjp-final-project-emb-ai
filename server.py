"""Flask server for emotion detection API."""

from flask import Flask, jsonify, request

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/emotionDetector", methods=["POST"])
def detect_emotion():
    """
    Handle POST requests to /emotionDetector.

    Expects JSON body: {"text": "I love my life"}
    Returns JSON: {"response": "<formatted message>"}

    If dominant_emotion is None (e.g., blank input), returns:
    {"response": "Invalid text! Please try again!"}
    """
    data = request.get_json() or {}
    text = data.get("text", "")

    result = emotion_detector(text)

    if result.get("dominant_emotion") is None:
        return jsonify({"response": "Invalid text! Please try again!"})

    response_text = (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} "
        f"and 'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )

    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    