import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Games', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Plot scores
    plt.plot(scores, label='Score', color='blue', linestyle='-')
    
    # Plot mean scores
    plt.plot(mean_scores, label='Mean Score', color='red', linestyle='-')
    
    # Last score
    plt.scatter(len(scores)-1, scores[-1], color='blue', marker='o', s=50)
    plt.scatter(len(mean_scores)-1, mean_scores[-1], color='red', marker='o', s=50)
    
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), fontsize=10, verticalalignment='bottom')
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), fontsize=10, verticalalignment='bottom')
    
    plt.legend(loc='upper left', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Y-axis limit
    max_score = max(max(scores), max(mean_scores))
    plt.ylim(0, max_score + max_score * 0.1)
    
    # Set axis limits
    plt.xlim(0, max(len(scores), len(mean_scores)))
    
    plt.show(block=False)
    plt.pause(0.1)
