import matplotlib.pyplot as plt

def plot_metrics(metrics_dict):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.bar(names, values)
    plt.title("Model Performance")
    plt.show()