from translator.translator_service import TranslatorService


def main():
    translator_service = TranslatorService()
    n_epochs=3
    seed=42
    translator_service.handle_translations(n_epochs, seed)
  
if __name__ == "__main__":
    main()