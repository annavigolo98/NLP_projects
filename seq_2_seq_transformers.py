from transformer.seq2seq_service import Seq2SeqService

#!pip install transformers datasets sentencepiece sacremoses

def main():

    service = Seq2SeqService()
    service.handle_seq2seq()

if __name__ == "__main__":
    main()