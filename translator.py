from transformer.seq2seq_service import Seq2SeqService

#!pip install transformers datasets sentencepiece sacremoses

def main():

    service = Seq2SeqService()
    sentence_to_translate = 'You cannot make it! My big fail fuck!'
    service.translate_sentence(sentence_to_translate)



if __name__ == "__main__":
    main()