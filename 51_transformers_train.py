from transformer.seq2seq_service import Seq2SeqService

def main():

    seed = 42
    n_epochs=4

    service = Seq2SeqService()
    service.handle_seq2seq(seed, n_epochs)

if __name__ == "__main__":
    main()