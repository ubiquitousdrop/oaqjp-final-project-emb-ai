import requests


def emotion_detector(text_to_analyze: str) -> dict:
    """
    Sends text to Watson NLP EmotionPredict endpoint and returns a dictionary with
    anger, disgust, fear, joy, sadness scores and the dominant_emotion.

    Output format:
    {
        'anger': <float>,
        'disgust': <float>,
        'fear': <float>,
        'joy': <float>,
        'sadness': <float>,
        'dominant_emotion': '<name of the dominant emotion>'
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