from NER.ner_eval_service import NEREvalService
 

def main():
    NER_eval_service = NEREvalService()
    sentence_to_NER = 'Bill Gates was the CEO of Microsoft in Seattle, Washington.'
    NER_eval_service.evaluate_NER(sentence_to_NER)
  
if __name__ == "__main__":
    main()