import matplotlib.pyplot as plt
import numpy as np 
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Calculate standard deviation
    std_deviation = np.std(scores)
    
    # Include standard deviation in the plot title
    plt.title(f'Training Progress - σ: {std_deviation:.2f}', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Games', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Plot scores
    plt.plot(scores, label='Score', color='blue', linestyle='-')
    
    # Plot mean scores
    plt.plot(mean_scores, label='Mean Score', color='red', linestyle='-')
    
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def plot_two(score_one, score_two, mean_score_one, mean_score_two):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    # Calculate standard deviation
    std_deviation_one = np.std(score_one)
    std_deviation_two = np.std(score_two)
    
    # Include standard deviation in the plot title
    plt.title(f'Training Progress - σ: {std_deviation_one:.2f} , {std_deviation_two:.2f}', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Games', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Plot scores
    plt.plot(score_one, label='Score 1', color='blue', linestyle='-')
    plt.plot(score_two, label='Score 2', color='red', linestyle='-')
    
    # Plot mean scores
    plt.plot(mean_score_one, label='Mean Score 1', color='darkblue', linestyle='-')
    plt.plot(mean_score_two, label='Mean Score 2', color='darkred', linestyle='-')
    
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()