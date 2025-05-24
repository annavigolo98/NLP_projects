from transformer.seq2seq_service import Seq2SeqService

def main():

    service = Seq2SeqService()
    sentence_to_translate = 'I love pizza'
    service.translate_sentence(sentence_to_translate)



if __name__ == "__main__":
    main()