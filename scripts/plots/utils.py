import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="whitegrid", rc=custom_params)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 15,  # Set font size to 11pt
            "axes.labelsize": 15,  # -> axis labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "text.usetex": False,
            "pgf.rcfonts": False,
        }
    )


def get_elongated_size(multiplier=2):
    return [6.4 * multiplier, 4.8]


def elongate_plot(multiplier=2):
    plt.rcParams.update(
        {
            "figure.figsize": get_elongated_size(multiplier),
        }
    )


def flatten_plots():
    plt.rcParams.update(
        {
            "figure.figsize": [6.4, 3.2],
        }
    )


def get_flattened_figsize(num_subplots=[1, 1]):
    return [6.4 * num_subplots[0], 3.2 * num_subplots[1]]


def flatten_plots_higher():
    plt.rcParams.update(
        {
            "figure.figsize": [6.4, 4.4],
        }
    )


models_names = {
    "llama3_8b_inst": "Llama3-8B",
    "mistral12b_inst": "Mistral-12B",
    "mistral7b_inst": "Mistral-7B",
    "olmo2_1124_7b_inst": "OLMo2-7B",
    "olmo2_1124_13b_inst": "OLMo2-13B",
    "qwen3_8b": "Qwen3-8B",
}

color_palette = {
    "Llama3-8B": "#0568e2",
    "Mistral-7B": "#ff7001",
    "OLMo2-7B": "#f0529c",
    "Qwen3-8B": "#8b33ff",
}
