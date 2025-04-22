
from NER.ner_custom_eval_service import NERCustomEvalService


def main():

    string = 'I have been living in Germany for 6 months.'
    ner_custom_eval_service = NERCustomEvalService()
    ner_custom_eval_service.evaluate_custom_ner_dataset(string)


if __name__ == "__main__":
    main()