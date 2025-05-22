from transformer.seq2seq_service import Seq2SeqService

def main():
    
    n_epochs=6
    seed = 98
    service = Seq2SeqService()
    service.handle_seq2seq(n_epochs, seed)

if __name__ == "__main__":
    main()