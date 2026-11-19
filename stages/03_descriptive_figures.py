
# ============================================================
# 5. Figure-generation utilities and exploratory figures
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 500,
})

figure_records = []

def save_fig(filename):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches="tight")
    figure_records.append({"Figure": filename, "Path": str(path)})
    plt.close()
    return path

# Figure 1: PCE distribution
plt.figure(figsize=(6.4, 4.2))
plt.hist(df["PCE"], bins=40, edgecolor="black", linewidth=0.5)
plt.xlabel("Power conversion efficiency, PCE (%)")
plt.ylabel("Number of configurations")
plt.title("Distribution of SCAPS-1D simulated PCE")
save_fig("Figure_01_PCE_Distribution.jpg")

# Figure 2: ETL/HTL heatmap based on maximum PCE
heatmap_data = etl_htl_ranking.pivot(index="HTL", columns="ETL", values="PCE_max")
plt.figure(figsize=(6.2, 4.5))
im = plt.imshow(heatmap_data.values, aspect="auto")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.colorbar(im, label="Maximum PCE (%)")
plt.xlabel("ETL")
plt.ylabel("HTL")
plt.title("ETL/HTL influence on maximum PCE")
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.iloc[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
save_fig("Figure_02_ETL_HTL_Max_PCE_Heatmap.jpg")

# Figure 3: Top 15 configurations by SCAPS PCE
plot_top = top_50_by_scaps_pce.head(15).copy()
plot_top["short_label"] = plot_top["ETL"] + "/" + plot_top["HTL"] + " | " + plot_top["top_absorber_thickness_um"].astype(str) + "/" + plot_top["bottom_absorber_thickness_um"].astype(str) + " um"
plt.figure(figsize=(8.5, 5.8))
y_pos = np.arange(len(plot_top))[::-1]
plt.barh(y_pos, plot_top["PCE"].values[::-1])
plt.yticks(y_pos, plot_top["short_label"].values[::-1])
plt.xlabel("PCE (%)")
plt.title("Top 15 SCAPS-1D configurations by PCE")
save_fig("Figure_03_Top_15_SCAPS_PCE_Configurations.jpg")

# Figure 4: Parameter effects on maximum PCE
for group_col, fig_name, x_label in [
    ("top_absorber_thickness_um", "Figure_04_Top_Absorber_Thickness_Effect.jpg", "Top absorber thickness (um)"),
    ("bottom_absorber_thickness_um", "Figure_05_Bottom_Absorber_Thickness_Effect.jpg", "Bottom absorber thickness (um)"),
    ("top_absorber_bandgap_ev", "Figure_06_Top_Absorber_Bandgap_Effect.jpg", "Top absorber bandgap (eV)"),
    ("bottom_absorber_bandgap_ev", "Figure_07_Bottom_Absorber_Bandgap_Effect.jpg", "Bottom absorber bandgap (eV)"),
    ("log10_top_absorber_doping", "Figure_08_Top_Doping_Effect.jpg", "log10(top absorber doping, cm^-3)"),
    ("log10_bottom_absorber_acceptor", "Figure_09_Bottom_Acceptor_Effect.jpg", "log10(bottom absorber acceptor density, cm^-3)"),
]:
    temp = df.groupby(group_col)["PCE"].agg(["mean", "max", "std"]).reset_index()
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(temp[group_col], temp["mean"], marker="o", label="Mean PCE")
    plt.plot(temp[group_col], temp["max"], marker="s", label="Maximum PCE")
    plt.xlabel(x_label)
    plt.ylabel("PCE (%)")
    plt.title(x_label + " effect on PCE")
    plt.legend()
    save_fig(fig_name)

figure_records_df = pd.DataFrame(figure_records)
display(figure_records_df)
