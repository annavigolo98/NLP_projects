import numpy as np
from pydantic import BaseModel 
import matplotlib.pyplot as plt

class Plotter(BaseModel):
    def plot_loss(self, train_loss: np.ndarray, validation_loss: np.ndarray):
        plt.plot(train_loss, label='Train loss')
        plt.plot(validation_loss, label='Validation loss')
        plt.title('Loss vs number of epochs')
        plt.xlabel('N_epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig('transformer/plots/loss_vs_epoch_plot.png')
        plt.close()


    def plot_metrics(self, train_metric: np.ndarray, validation_metric: np.ndarray, metric_name: str):
        plt.plot(train_metric, label='Train metric')
        plt.plot(validation_metric, label='Validation metric')
        plt.title(f'{metric_name} metric vs number of epochs')
        plt.xlabel('N_epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(f'transformer/plots/{metric_name}_vs_epoch_plot.png')
        plt.close()