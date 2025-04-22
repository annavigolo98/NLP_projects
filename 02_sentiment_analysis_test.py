from sentiments.sentiment_evaluation_service import SentimentEvaluationService


def main():
    sentence = 'This film is spannend! Really good job!'
    sentiment_evaluation_service = SentimentEvaluationService()
    sentiment_evaluation_service.evaluate_sentiments(sentence)
  
if __name__ == "__main__":
    main()