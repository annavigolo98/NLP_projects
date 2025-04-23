from NER.ner_service import NERService
 

def main():
    n_epochs = 3
    NER_service = NERService()
    NER_service.handle_NER(n_epochs)
  
if __name__ == "__main__":
    main()