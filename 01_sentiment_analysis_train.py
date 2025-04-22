from sentiments.sentiment_analysis_service import SentimentAnalysisService


def main():
    sentiment_analysis_service = SentimentAnalysisService()
    sentiment_analysis_service.handle_sentiment_analysis()
  
if __name__ == "__main__":
    main()