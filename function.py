import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def conf_matrix(y, y_pred, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    labels = [
        "EN",
        "Negative",
        "Neutral",
        "Positive",
        "EP",
    ]
    ax = sns.heatmap(
        confusion_matrix(y, y_pred),
        annot=True,
        cmap="Blues",
        fmt="g",
        cbar=False,
        annot_kws={"size": 25},
    )
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17)
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel("Test", fontsize=20)
    ax.set_xlabel("Predicted", fontsize=20)
    plt.show()
