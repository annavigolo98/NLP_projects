
from NER.ner_custom_service import NERCustomService


def main():
    ner_custom_service = NERCustomService()
    seed = 42
    n_epochs = 1
    ner_custom_service.handle_custom_ner_dataset(n_epochs, seed)


if __name__ == "__main__":
    main()





