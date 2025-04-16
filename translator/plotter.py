from pydantic import BaseModel
import matplotlib.pyplot as plt 

class Plotter(BaseModel):
    def plot_len_sentence_histogram(self, dataset, choice):
        
        train = dataset['train']['translation']
        if choice == 'en':
            input_lens = [len(tr['en']) for tr in train]

            plt.hist(input_lens, bins=50)
            plt.title('English sentences histogram train')
            plt.xlabel('lenght of the sentence')
            plt.ylabel('frequency')
            plt.show()
        if choice == 'fr':
            input_lens = [len(tr['fr']) for tr in train]

            plt.hist(input_lens, bins=50)
            plt.title('French sentences histogram train')
            plt.xlabel('lenght of the sentence')
            plt.ylabel('frequency')
            plt.show()