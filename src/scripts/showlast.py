import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("./runs/run_last.csv")

    # cols = ["mse", "mae", "r2", "rmse", "mad_outlier_fraction", "local_mad_outlier_fraction"]
    # cols = ["mse", "mae", "r2", "kurtosis", "skewness", "fitted_outlier_fraction"]
    # sns.pairplot(df, hue="algorithm", diag_kind="kde", plot_kws={'alpha':0.5}, vars=cols)
    # plt.show()

    fig, axes = plt.subplots(1,4, figsize=(12, 4))
    sns.boxplot(data=df, x="algorithm", y="fitted_outlier_fraction", ax=axes[0])
    sns.boxplot(data=df, x="algorithm", y="kurtosis", ax=axes[1])
    sns.boxplot(data=df, x="algorithm", y="skewness", ax=axes[2])
    sns.boxplot(data=df, x="algorithm", y="mSSIM", ax=axes[3])

    plt.show()