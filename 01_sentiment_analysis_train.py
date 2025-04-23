from sentiments.sentiment_analysis_service import SentimentAnalysisService


def main():
    sentiment_analysis_service = SentimentAnalysisService()
    n_epochs = 5
    sentiment_analysis_service.handle_sentiment_analysis(n_epochs)
  
if __name__ == "__main__":
    main()