from translator.translator_eval_service import TranslatorEvalService 


def main():
    sentence='Translate this sentence into French please.'
    translator_eval_service = TranslatorEvalService()
    translator_eval_service.evaluate_translator(sentence)
  
if __name__ == "__main__":
    main()