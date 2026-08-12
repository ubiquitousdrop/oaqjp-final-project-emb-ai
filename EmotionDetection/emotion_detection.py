import requests


def emotion_detector(text_to_analyze: str) -> dict:
    """
    Sends text to Watson NLP EmotionPredict endpoint and returns a dictionary with
    anger, disgust, fear, joy, sadness scores and the dominant_emotion.

    If the server returns status code 400 (e.g., blank input), returns a dictionary
    with all values set to None.

    Output format:
    {
        'anger': <float or None>,
        'disgust': <float or None>,
        'fear': <float or None>,
        'joy': <float or None>,
        'sadness': <float or None>,
        'dominant_emotion': <str or None>
    }
    """
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    # Handle blank input / bad request
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response.raise_for_status()

    data = response.json()

    # Extract emotion scores from the first prediction's first mention
    emotions = data["emotionPredictions"][0]["emotionMentions"][0]["emotion"]

    anger = emotions["anger"]
    disgust = emotions["disgust"]
    fear = emotions["fear"]
    joy = emotions["joy"]
    sadness = emotions["sadness"]

    # Find dominant emotion (highest score)
    emotion_scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }