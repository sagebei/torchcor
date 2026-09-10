import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

DATA_PATH = Path(__file__).parent / "data"


def load_dataset_from_pickle(filename: str | Path) -> dict[str, np.ndarray]:
    if not isinstance(filename, Path):
        filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")

    with open(filename.as_posix(), "rb") as fl:
        return pickle.load(fl)


LABEL_NAMES = [
    "CARPentry",
    "Ambit",
    "4C",
    "Simula",
    "CHimeRA",
    r"$\mathcal{C}$Heart",
    r"life$^{\mathbf{X}}$",
    r"SimVascular $\mathbb{P}_1$",
    r"SimVascular $\mathbb{P}_2$",
    "COMSOL",
]

LABEL_NAMES_BIV = [
    "CARPentry",
    "Ambit",
    "4C",
    "Simula",
    "CHimeRA",
    r"$\mathcal{C}$Heart",
    r"life$^{\mathbf{X}}$",
    r"SimVascular $\mathbb{P}_1$",
    "COMSOL",
]

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:cyan",
    "tab:olive",
]

COLORS_BIV = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
]

LABELS_P1 = [
    r"Simula-$\mathbb{P}_1$",
    r"CHimeRA-$\mathbb{P}_1$",
    r"$\mathcal{C}$Heart-$\mathbb{P}_1$",
    r"life$^{\mathbf{X}}$-$\mathbb{P}_1$",
    r"COMSOL-$\mathbb{P}_1$",
]

LABELS_P2 = [
    r"Simula-$\mathbb{P}_2$",
    r"CHimeRA-$\mathbb{P}_2$",
    r"$\mathcal{C}$Heart-$\mathbb{P}_2$",
    r"life$^{\mathbf{X}}$-$\mathbb{P}_2$",
    r"COMSOL-$\mathbb{P}_2$",
]

COLORS_P1_P2 = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]

TEAMS_DATASETS_0A = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_carpentry.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_ambit.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_4C.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_simula.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_chimera.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_cheart.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_lifex.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_simvascular_p1p1.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_simvascular_p2.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0A_group_comsol.pickle"
    ),
]

TEAMS_DATASET_0B = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_carpentry.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_ambit.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_4C.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_simula.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_chimera.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_cheart.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_lifex.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_simvascular_p1p1.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_simvascular_p2.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_0B_group_comsol.pickle"
    ),
]

TEAMS_DATASETS_1 = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_carpentry.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_ambit.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_4c.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_simula.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_chimera.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_cheart.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_lifex.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_simvascular_p1p1.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_simvascular_p2.pickle"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_nonblinded_step_1_group_comsol.pickle"
    ),
]

TEAMS_DATASETS_A = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_A_group_carpentry.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_ambit.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_4c.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_lifex.pkl"),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_A_group_simvascular_p1p1.pkl"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_A_group_simvascular_p2.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_A_group_comsol.pkl"),
]

TEAMS_DATASETS_B = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_B_group_carpentry.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_ambit.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_4c.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_lifex.pkl"),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_B_group_simvascular_p1p1.pkl"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_B_group_simvascular_p2.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_B_group_comsol.pkl"),
]
TEAMS_DATASETS_C = [
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_C_group_carpentry.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_ambit.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_4c.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_lifex.pkl"),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_C_group_simvascular_p1p1.pkl"
    ),
    load_dataset_from_pickle(
        DATA_PATH / "monoventricular_blinded_C_group_simvascular_p2.pkl"
    ),
    load_dataset_from_pickle(DATA_PATH / "monoventricular_blinded_C_group_comsol.pkl"),
]

TEAMS_DATASETS_BIV_COARSE_P2 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_carpentry.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_ambit.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_4c.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_lifex.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_simvascular.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_comsol.pkl"),
]

TEAMS_DATASETS_BIV_FINE_P2 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_carpentry.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_ambit.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_4c.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_lifex.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_simvascular.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_comsol.pkl"),
]


REDUCED_TEAMS_DATASETS_BIV_COARSE_P1 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_simula_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_chimera_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_cheart_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_lifex_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_comsol_p1.pkl"),
]
REDUCED_TEAMS_DATASETS_BIV_COARSE_P2 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_lifex.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_coarse_group_comsol.pkl"),
]

REDUCED_TEAMS_DATASETS_BIV_FINE_P1 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_simula_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_chimera_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_cheart_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_lifex_p1.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_comsol_p1.pkl"),
]
REDUCED_TEAMS_DATASETS_BIV_FINE_P2 = [
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_simula.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_chimera.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_cheart.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_lifex.pkl"),
    load_dataset_from_pickle(DATA_PATH / "biventricular_fine_group_comsol.pkl"),
]


def compute_plot_displacement_monoventricular(
    teams_datasets: list[dict[str, np.ndarray]],
    filename: str | Path,
    labels_names: list[str],
    colors: list[str],
) -> None:
    """Computes the displacement figures based on the groups datasets."""
    if isinstance(filename, str):
        filename = Path(filename)

    if not isinstance(teams_datasets, list):
        raise Exception("teams_datasets must be a list")

    if not len(teams_datasets) == len(labels_names):
        raise Exception("teams_datasets and labels_names must have the same length")

    if not len(teams_datasets) == len(colors):
        raise Exception("teams_datasets and colors must have the same length")

    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 12))

    axs[0, 0].set_title(r"Particle $\mathbf{p}_0$")
    axs[0, 0].set_ylabel("Displacement x-component [m]")
    axs[1, 0].set_ylabel("Displacement y-component [m]")
    axs[2, 0].set_ylabel("Displacement z-component [m]")
    axs[2, 0].set_xlabel("Time [s]")

    axs[0, 1].set_title(r"Particle $\mathbf{p}_1$")
    axs[2, 1].set_xlabel("Time [s]")

    for data, lbl, color in zip(teams_datasets, labels_names, colors):
        for i, u_type in zip([0, 1, 2], ["ux", "uy", "uz"]):
            axs[i, 0].plot(
                data["time"], data["displacement"]["p0"][u_type], label=lbl, color=color
            )
            axs[i, 1].plot(
                data["time"], data["displacement"]["p1"][u_type], label=lbl, color=color
            )

    plt.legend(loc="best", fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(filename.as_posix(), bbox_inches="tight", dpi=120)
    # plt.show()


def compute_plot_displacements_biventricular(
    teams_datasets: list[dict[str, np.ndarray]],
    filename: str | Path,
    labels_names: list[str],
    colors: list[str],
) -> None:
    """Computes the displacement figures based on groups processed datasets"""
    if isinstance(filename, str):
        filename = Path(filename)

    if not isinstance(teams_datasets, list):
        raise Exception("teams_datasets must be a list")

    if not len(teams_datasets) == len(labels_names):
        raise Exception("teams_datasets and labels_names must have the same length")

    if not len(teams_datasets) == len(colors):
        raise Exception("teams_datasets and colors must have the same length")

    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(18, 12))

    axs[0, 0].set_title(r"Particle $\mathbf{p}_0$")
    axs[0, 0].set_ylabel("Displacement x-component [m]")
    axs[1, 0].set_ylabel("Displacement y-component [m]")
    axs[2, 0].set_ylabel("Displacement z-component [m]")
    axs[2, 0].set_xlabel("Time [s]")

    axs[0, 1].set_title(r"Particle $\mathbf{p}_1$")
    axs[0, 2].set_title(r"Particle $\mathbf{p}_2$")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 2].set_xlabel("Time [s]")

    for data, lbl, color in zip(teams_datasets, labels_names, colors):
        for i, u_type in zip([0, 1, 2], ["ux", "uy", "uz"]):
            axs[i, 0].plot(
                data["time"], data["displacement"]["p0"][u_type], label=lbl, color=color
            )
            axs[i, 1].plot(
                data["time"], data["displacement"]["p1"][u_type], label=lbl, color=color
            )
            axs[i, 2].plot(
                data["time"], data["displacement"]["p2"][u_type], label=lbl, color=color
            )

    axs[2, 2].legend(loc="best", fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(filename.as_posix(), bbox_inches="tight", dpi=120)
    # plt.show()


def compute_plot_displacements_p1_vs_p2_biventricular(
    teams_datasets_p1: list[dict[str, np.ndarray]],
    teams_datasets_p2: list[dict[str, np.ndarray]],
    filename: str | Path,
    labels_names_p1: list[str],
    labels_names_p2: list[str],
    colors: list[str],
) -> None:
    """Computes the displacement figures based on groups processed datasets"""
    if isinstance(filename, str):
        filename = Path(filename)

    if not isinstance(teams_datasets_p1, list) or not isinstance(
        teams_datasets_p2, list
    ):
        raise Exception("teams_datasets must be a list")

    if not len(teams_datasets_p1) == len(labels_names_p1):
        raise Exception("teams_datasets and labels_names must have the same length")

    if not len(teams_datasets_p2) == len(labels_names_p2):
        raise Exception("teams_datasets and labels_names must have the same length")

    if not len(labels_names_p1) == len(colors):
        raise Exception("teams_datasets and colors must have the same length")

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(18, 12))
    axs[0, 0].set_title(r"Particle $\mathbf{p}_0$")
    axs[0, 0].set_ylabel("Displacement x-component [m]")
    axs[1, 0].set_ylabel("Displacement y-component [m]")
    axs[2, 0].set_ylabel("Displacement z-component [m]")
    axs[2, 0].set_xlabel("Time [s]")

    axs[0, 1].set_title(r"Particle $\mathbf{p}_1$")
    axs[0, 2].set_title(r"Particle $\mathbf{p}_2$")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 2].set_xlabel("Time [s]")

    for data, lbl, data_p2, lbl_p2, color in zip(
        teams_datasets_p1, labels_names_p1, teams_datasets_p2, labels_names_p2, colors
    ):
        for i, u_type in zip([0, 1, 2], ["ux", "uy", "uz"]):
            axs[i, 0].plot(
                data["time"],
                data["displacement"]["p0"][u_type],
                label=lbl,
                linestyle="dashed",
                color=color,
            )
            axs[i, 1].plot(
                data["time"],
                data["displacement"]["p1"][u_type],
                label=lbl,
                linestyle="dashed",
                color=color,
            )
            axs[i, 2].plot(
                data["time"],
                data["displacement"]["p2"][u_type],
                label=lbl,
                linestyle="dashed",
                color=color,
            )

            axs[i, 0].plot(
                data_p2["time"],
                data_p2["displacement"]["p0"][u_type],
                label=lbl_p2,
                color=color,
            )
            axs[i, 1].plot(
                data_p2["time"],
                data_p2["displacement"]["p1"][u_type],
                label=lbl_p2,
                color=color,
            )
            axs[i, 2].plot(
                data_p2["time"],
                data_p2["displacement"]["p2"][u_type],
                label=lbl_p2,
                color=color,
            )

    axs[2, 2].legend(loc="best", fancybox=True, shadow=True)
    fig.tight_layout()
    fig.savefig(filename.as_posix(), bbox_inches="tight", dpi=120)
    # plt.show()


def compute_statistics(
    datasets: list[dict[str, np.ndarray]], point: str = "p0"
) -> dict[str, np.ndarray]:
    time = datasets[0]["time"]
    number_of_element_per_dataset = datasets[0]["time"].shape[0]
    number_of_datasets = len(datasets)
    condensated_dataset_ux = np.zeros(
        (number_of_element_per_dataset, number_of_datasets)
    )
    condensated_dataset_uy = np.zeros_like(condensated_dataset_ux)
    condensated_dataset_uz = np.zeros_like(condensated_dataset_ux)
    red_dataset_u = np.zeros((number_of_datasets,))

    for i, dataset in enumerate(datasets):
        condensated_dataset_ux[:, i] = dataset["displacement"][point]["ux"]
        condensated_dataset_uy[:, i] = dataset["displacement"][point]["uy"]
        condensated_dataset_uz[:, i] = dataset["displacement"][point]["uz"]

    mean_ux = np.mean(condensated_dataset_ux, axis=1)
    mean_uy = np.mean(condensated_dataset_uy, axis=1)
    mean_uz = np.mean(condensated_dataset_uz, axis=1)

    for i in range(number_of_datasets):
        diff_to_mean_norm = np.sqrt(
            np.abs(condensated_dataset_ux[:, i] - mean_ux) ** 2
            + np.abs(condensated_dataset_uy[:, i] - mean_uy) ** 2
            + np.abs(condensated_dataset_uz[:, i] - mean_uz) ** 2
        )

        mean_norm = np.sqrt(
            np.abs(mean_ux) ** 2 + np.abs(mean_uy) ** 2 + np.abs(mean_uz) ** 2
        )

        red_dataset_u[i] = np.mean(diff_to_mean_norm / mean_norm, axis=0)

    statistics_dataset = {
        "time": time,
        "mean_ux": mean_ux,
        "mean_uy": mean_uy,
        "mean_uz": mean_uz,
        "std_ux": np.std(condensated_dataset_ux, axis=1),
        "std_uy": np.std(condensated_dataset_uy, axis=1),
        "std_uz": np.std(condensated_dataset_uz, axis=1),
        "red_u": red_dataset_u,
    }

    return statistics_dataset


def compute_plots_biventricular_blinded_step() -> None:
    """Computes the displacement curves for the biventricular blinded step"""
    compute_plot_displacements_p1_vs_p2_biventricular(
        REDUCED_TEAMS_DATASETS_BIV_COARSE_P1,
        REDUCED_TEAMS_DATASETS_BIV_COARSE_P2,
        "./comparison_plots_space_step_3_blinded_coarse.png",
        LABELS_P1,
        LABELS_P2,
        COLORS_P1_P2,
    )
    compute_plot_displacements_p1_vs_p2_biventricular(
        REDUCED_TEAMS_DATASETS_BIV_FINE_P1,
        REDUCED_TEAMS_DATASETS_BIV_FINE_P2,
        "./comparison_plots_space_step_3_blinded_fine.png",
        LABELS_P1,
        LABELS_P2,
        COLORS_P1_P2,
    )

    compute_plot_displacements_biventricular(
        TEAMS_DATASETS_BIV_COARSE_P2,
        "./comparison_plots_p0_p1_step_3_blinded_coarse.png",
        LABEL_NAMES_BIV,
        COLORS_BIV,
    )
    compute_plot_displacements_biventricular(
        TEAMS_DATASETS_BIV_FINE_P2,
        "./comparison_plots_p0_p1_step_3_blinded_fine.png",
        LABEL_NAMES_BIV,
        COLORS_BIV,
    )


def compute_plots_monoventricular_nonblinded_step_0() -> None:
    """Compute displacement curves for the monoventricular non-blinded step 0"""
    compute_plot_displacement_monoventricular(
        TEAMS_DATASETS_0A,
        "./comparison_plots_p0_p1_step_0A_nonblinded.png",
        LABEL_NAMES,
        COLORS,
    )
    compute_plot_displacement_monoventricular(
        TEAMS_DATASET_0B,
        "./comparison_plots_p0_p1_step_0B_nonblinded.png",
        LABEL_NAMES,
        COLORS,
    )


def compute_plots_monoventricular_nonblinded_step_1() -> None:
    """Computes displacement curves for the monoventricular non-blinded step 1"""
    compute_plot_displacement_monoventricular(
        TEAMS_DATASETS_1,
        "./comparison_plots_p0_p1_step_1_nonblinded.png",
        LABEL_NAMES,
        COLORS,
    )


def compute_statistics_per_dataset(
    labels: list[str], datasets: list[dict[str, np.ndarray]], header: str = ""
) -> None:
    if len(labels) != len(datasets):
        raise ValueError("labels and datasets do not match in length")

    print(header)
    statistics_p0 = compute_statistics(datasets, point="p0")
    statistics_p1 = compute_statistics(datasets, point="p1")

    red_p0 = np.round(statistics_p0["red_u"], decimals=3)
    red_p1 = np.round(statistics_p1["red_u"], decimals=3)

    print(f"Team {'TEAM NAME':25s} - {'p0':4s}, {'p1':4s}")
    for label, i in zip(labels, range(len(red_p0))):
        print(f"Team {label:25s} - {red_p0[i]:4.3f}, {red_p1[i]:4.3f}")

    print("-----------------------------------\n")


def compute_plots_monoventricular_blinded() -> None:
    """Computes displacement curves for the monoventricular blinded cases A, B, C"""

    compute_plot_displacement_monoventricular(
        TEAMS_DATASETS_A,
        "./comparison_plots_p0_p1_step_2A_blinded.png",
        LABEL_NAMES,
        COLORS,
    )
    compute_plot_displacement_monoventricular(
        TEAMS_DATASETS_B,
        "./comparison_plots_p0_p1_step_2B_blinded.png",
        LABEL_NAMES,
        COLORS,
    )
    compute_plot_displacement_monoventricular(
        TEAMS_DATASETS_C,
        "./comparison_plots_p0_p1_step_2C_blinded.png",
        LABEL_NAMES,
        COLORS,
    )
    compute_statistics_per_dataset(
        LABEL_NAMES, TEAMS_DATASETS_A, header="RED Computation, Case A"
    )
    compute_statistics_per_dataset(
        LABEL_NAMES, TEAMS_DATASETS_B, header="RED Computation, Case B"
    )
    compute_statistics_per_dataset(
        LABEL_NAMES, TEAMS_DATASETS_C, header="RED Computation, Case C"
    )


if __name__ == "__main__":
    compute_plots_monoventricular_nonblinded_step_0()
    compute_plots_monoventricular_nonblinded_step_1()
    compute_plots_monoventricular_blinded()
    compute_plots_biventricular_blinded_step()
