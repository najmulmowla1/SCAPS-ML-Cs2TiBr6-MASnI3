
# ============================================================
# 3. Descriptive statistics and best SCAPS configuration
# ============================================================

input_features_raw = [
    "ETL", "HTL",
    "top_absorber_thickness_um", "bottom_absorber_thickness_um",
    "top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev",
    "top_absorber_doping_cm3", "bottom_absorber_acceptor_cm3",
]

target_cols = ["Voc", "Jsc", "FF", "PCE"]

numeric_feature_cols = [
    "top_absorber_thickness_um", "bottom_absorber_thickness_um",
    "top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev",
    "top_absorber_doping_cm3", "bottom_absorber_acceptor_cm3",
    "log10_top_absorber_doping", "log10_bottom_absorber_acceptor",
]

dataset_summary = pd.DataFrame({
    "Metric": [
        "Original rows", "Clean valid rows", "Removed invalid/empty rows",
        "Number of ETL materials", "Number of HTL materials",
        "Device architecture", "Target outputs"
    ],
    "Value": [
        len(raw_df), len(df), len(raw_df) - len(df),
        df["ETL"].nunique(), df["HTL"].nunique(),
        "FTO/ETL/Cs2TiBr6/MASnI3/HTL/Au", "Voc, Jsc, FF, PCE"
    ]
})

input_parameter_ranges = []
for col in numeric_feature_cols:
    input_parameter_ranges.append({
        "Parameter": col,
        "Minimum": df[col].min(),
        "Maximum": df[col].max(),
        "Mean": df[col].mean(),
        "Std": df[col].std(),
        "Unique_values": df[col].nunique()
    })
input_parameter_ranges = pd.DataFrame(input_parameter_ranges)

output_statistics = df[target_cols].describe().T.reset_index().rename(columns={"index": "Output"})

best_overall = df.head(1).copy()
top_50_by_scaps_pce = df.head(50).copy()

print("Dataset summary:")
display(dataset_summary)

print("Input parameter ranges:")
display(input_parameter_ranges)

print("Output statistics:")
display(output_statistics)

print("Best SCAPS-1D configuration by PCE:")
display(best_overall[["configuration", "ETL", "HTL", "top_absorber_thickness_um", "bottom_absorber_thickness_um",
                      "top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev", "top_absorber_doping_cm3",
                      "bottom_absorber_acceptor_cm3", "Voc", "Jsc", "FF", "PCE"]])



# ============================================================
# 4. Physics-informed grouping analysis
# ============================================================

def summarize_by(group_cols, data=df, targets=target_cols):
    agg_dict = {}
    for t in targets:
        agg_dict[f"{t}_mean"] = (t, "mean")
        agg_dict[f"{t}_std"] = (t, "std")
        agg_dict[f"{t}_min"] = (t, "min")
        agg_dict[f"{t}_max"] = (t, "max")
    out = data.groupby(group_cols).agg(**agg_dict, count=("PCE", "size")).reset_index()
    return out.sort_values("PCE_max", ascending=False).reset_index(drop=True)

etl_ranking = summarize_by(["ETL"])
htl_ranking = summarize_by(["HTL"])
etl_htl_ranking = summarize_by(["ETL", "HTL"])
configuration_ranking = summarize_by(["configuration"])

top_thickness_effect = summarize_by(["top_absorber_thickness_um"])
bottom_thickness_effect = summarize_by(["bottom_absorber_thickness_um"])
top_bandgap_effect = summarize_by(["top_absorber_bandgap_ev"])
bottom_bandgap_effect = summarize_by(["bottom_absorber_bandgap_ev"])
top_doping_effect = summarize_by(["top_absorber_doping_cm3", "log10_top_absorber_doping"])
bottom_acceptor_effect = summarize_by(["bottom_absorber_acceptor_cm3", "log10_bottom_absorber_acceptor"])


thickness_interaction = summarize_by(["top_absorber_thickness_um", "bottom_absorber_thickness_um"])
bandgap_interaction = summarize_by(["top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev"])
doping_interaction = summarize_by(["top_absorber_doping_cm3", "bottom_absorber_acceptor_cm3"])

print("ETL/HTL ranking by maximum PCE:")
display(etl_htl_ranking.head(20))

print("Top-bandgap/bottom-bandgap interaction by maximum PCE:")
display(bandgap_interaction.head(20))

print("Top-thickness/bottom-thickness interaction by maximum PCE:")
display(thickness_interaction.head(20))
