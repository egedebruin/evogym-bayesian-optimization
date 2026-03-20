import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "b-f-r-results.csv"
SAMPLE_FRAC = 0.05

ENV_KEY_TO_LABEL = {
    'bidirectional': 'Bidirectional without sensor',
    'bidirectional2': 'Bidirectional with sensor',
}

def main():
    df = pd.read_csv(CSV_PATH)
    environments = df["environment"].unique()

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(
        nrows=len(environments),
        ncols=1,
        figsize=(5, 2.5 * len(environments)),  # ← smaller & tighter
        sharex=True
    )

    if len(environments) == 1:
        axes = [axes]

    for ax, env in zip(axes, environments):
        result_left = {}
        result_right = {}
        env_df = df[df["environment"] == env].copy()

        # --- keep sampling ---
        env_df = env_df.sample(frac=SAMPLE_FRAC, random_state=15)
        env_df = env_df.sort_values("value1", ascending=False).reset_index(drop=True)

        x = np.arange(len(env_df))
        y1 = env_df["value1"].values
        y2 = env_df["value2"].values

        result_left["cat"] = [f"{env}left" for _ in range(len(env_df))]
        result_left["x"] = [i + 1 for i in list(x)]
        result_left["y"] = y1
        result_left["ymin"] = [y if y < 0 else 0 for y in y1]
        result_left["ymax"] = [y if y > 0 else 0 for y in y1]
        pd.DataFrame(result_left).to_csv(f"{env}left.txt", index=False, sep="\t")

        result_right["cat"] = [f"{env}right" for _ in range(len(env_df))]
        result_right["x"] = [i + 1 for i in list(x)]
        result_right["y"] = y2
        result_right["ymin"] = [y if y < 0 else 0 for y in y2]
        result_right["ymax"] = [y if y > 0 else 0 for y in y2]
        pd.DataFrame(result_right).to_csv(f"{env}right.txt", index=False, sep="\t")

        c1 = "#4C72B0"
        c2 = "#DD8452"

        # Slightly smaller markers for compact layout
        ax.scatter(x, y1, s=22, color=c1, alpha=0.9)
        ax.scatter(x, y2, s=22, color=c2, alpha=0.9)

        ax.fill_between(x, y1, 0, color=c1, alpha=0.10)
        ax.fill_between(x, y2, 0, color=c2, alpha=0.10)

        ax.axhline(0, color="black", linewidth=0.9)

        ax.set_title(
            ENV_KEY_TO_LABEL[env],
            fontsize=13,
            weight="bold",
            pad=6
        )

        ax.set_ylabel("Fitness", fontsize=11)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel(
        "Robot–Controller pairs (sorted by left movement)",
        fontsize=11,
        labelpad=6
    )

    # Single legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor="#4C72B0", markersize=7, label='Left'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor="#DD8452", markersize=7, label='Right')
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=2, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    main()