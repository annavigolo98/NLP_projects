from transformer.seq2seq_service import Seq2SeqService

def main():
    
    n_epochs=4

    service = Seq2SeqService()
    service.handle_seq2seq(n_epochs)

if __name__ == "__main__":
    main()