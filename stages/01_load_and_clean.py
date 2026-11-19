
# ============================================================
# 2. Load and clean dataset
# ============================================================

raw_df = (pd.read_csv(DATASET_PATH) if DATASET_PATH.suffix.lower() == ".csv" else pd.read_excel(DATASET_PATH))
print("Raw dataset shape:", raw_df.shape)
print("Raw columns:")
print(raw_df.columns.tolist())

def normalize_column_name(col):
    col = str(col).strip()
    replacements = {
        "STRUCTURE": "structure",
        "TOP ABSORBER THICKNESS": "top_absorber_thickness_um",
        "BOTTOM ABSORBER THICKNESS": "bottom_absorber_thickness_um",
        "TOP ABSORBER BANDGAP": "top_absorber_bandgap_ev",
        "BOTTOM ABSORBER BANDGAP": "bottom_absorber_bandgap_ev",
        "T. A. DOPING DENSITY": "top_absorber_doping_cm3",
        "B. A. acceptor DENSITY": "bottom_absorber_acceptor_cm3",
        "Voc": "Voc",
        "Jsc": "Jsc",
        "FF": "FF",
        "PCE": "PCE",
    }
    return replacements.get(col, re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_").lower())

# Rename columns
df = raw_df.rename(columns={c: normalize_column_name(c) for c in raw_df.columns}).copy()

# Remove empty rows and invalid rows
critical_cols = ["structure", "top_absorber_thickness_um", "bottom_absorber_thickness_um",
                 "top_absorber_bandgap_ev", "bottom_absorber_bandgap_ev",
                 "top_absorber_doping_cm3", "bottom_absorber_acceptor_cm3",
                 "Voc", "Jsc", "FF", "PCE"]

for col in critical_cols:
    if col not in df.columns:
        raise KeyError(f"Required column not found after renaming: {col}")

# Convert numeric columns
numeric_cols = [c for c in critical_cols if c != "structure"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["structure"] = df["structure"].astype(str).str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=critical_cols).copy()

# Extract ETL and HTL materials from structure text
def extract_etl(structure):
    s = str(structure).lower().replace(" ", "")
    if "pcbm" in s:
        return "PCBM"
    if "c60" in s or "c₆₀" in s:
        return "C60"
    if "tio2" in s or "tio₂" in s:
        return "TiO2"
    return "Unknown_ETL"


def extract_htl(structure):
    s = str(structure).lower().replace(" ", "")
    if "gaas" in s:
        return "GaAs"
    if "moo3" in s or "moo₃" in s:
        return "MoO3"
    if "cu2o" in s or "cu₂o" in s:
        return "Cu2O"
    return "Unknown_HTL"


df["ETL"] = df["structure"].apply(extract_etl)
df["HTL"] = df["structure"].apply(extract_htl)

if (df["top_absorber_doping_cm3"] <= 0).any() or (df["bottom_absorber_acceptor_cm3"] <= 0).any():
    raise ValueError("Doping/acceptor densities must be positive for logarithmic transformation")
if (df["ETL"].str.startswith("Unknown_")).any() or (df["HTL"].str.startswith("Unknown_")).any():
    raise ValueError("Unknown transport-layer label(s) found in structure column. Review input structure naming")

# Add log-scaled density features for ML.
df["log10_top_absorber_doping"] = np.log10(df["top_absorber_doping_cm3"])
df["log10_bottom_absorber_acceptor"] = np.log10(df["bottom_absorber_acceptor_cm3"])

# Add compact configuration label for tables/figures
df["configuration"] = (
    "FTO/" + df["ETL"].astype(str) + "/Cs2TiBr6/MASnI3/" + df["HTL"].astype(str) + "/Au"
)

# Sort by PCE for reproducibility
df = df.sort_values("PCE", ascending=False).reset_index(drop=True)

if len(df) != 18225:
    print(f"WARNING: Expected 18,225 configurations for the published dataset, found {len(df)}. "
          "Continuing, but manuscript results will not be reproducible from this input.")
print("Cleaned dataset shape:", df.shape)
print("ETL materials:", sorted(df["ETL"].unique()))
print("HTL materials:", sorted(df["HTL"].unique()))
print("PCE range:", df["PCE"].min(), "to", df["PCE"].max())

display(df.head(10))
