import matplotlib.pyplot as plt
from utilities.model_utilities import load_object

def plot_encodings(names):
    fig, ax = plt.subplots()
    for name in names:
        encodings = load_object("./encodings/"+name)
        import numpy as np
        mean_array = np.mean(encodings, axis=0) 
        ax.scatter(range(mean_array.size), mean_array, label=name)
        # ax.plot(range(mean_array.size), mean_array, label=name)
    # leg = ax.legend(loc=2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=5, fancybox=True, shadow=True)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

def draw_neural_net(name, left=0.1, right=0.9, bottom=0.1, top=0.9):
    '''
    Modified from https://gist.github.com/craffel/2d727968c3aaebd10359
    '''

    fig, ax = plt.subplots()
    # mlp = load_object("./models/"+name)
    # print(mlp.coefs_)
    ax.axis('off')
    layer_sizes = [128,30,30,30,10] # TODO
    # n_layers = len(layer_sizes)
    n_layers = 5
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
    plt.show()

if __name__ == "__main__":
    # names = ["Ariana Grande", "Beyonce", "Chris Pratt", "Dwayne Johnson", "Justin Bieber", "Kim Kardashian", "Kylie Jenner", "Rihanna", "Selena Gomez", "Taylor Swift"]
    names = ["Ryan Gosling", "Emma Stone"]
    # plot_encodings(names)
    draw_neural_net("top10_mpl")