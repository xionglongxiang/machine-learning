import seaborn as sns
import matplotlib.pyplot as plt


def box_plot(train_data, feature):
    fig = plt.figure(figsize=(4,6))
    sns.boxplot(train_data[feature], orient='v', width=0.5)