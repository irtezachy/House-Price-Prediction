from sklearn import tree
import matplotlib.pyplot as plt
import os

def plot_tree(model, feature_names, save_path):
    plt.figure(figsize=(90, 30))
    tree.plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Tree plot saved to {os.path.abspath(save_path)}")
    else:
        plt.show()