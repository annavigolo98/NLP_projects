from translator.translator_service import TranslatorService


def main():
    translator_service = TranslatorService()
    seed=42
    translator_service.handle_translations(seed)
  
if __name__ == "__main__":
    main()