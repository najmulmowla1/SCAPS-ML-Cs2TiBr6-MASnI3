# Dataset schema expected by the supplied analysis notebook

The notebook originally reads `DATASET INITIALIZATION(1).xlsx` with these exact headers:

| Raw source column | Internal name | Units/meaning |
|---|---|---|
| `STRUCTURE` | `structure` | Full device architecture string; used to parse ETL and HTL |
| `TOP ABSORBER THICKNESS` | `top_absorber_thickness_um` | µm |
| `BOTTOM ABSORBER THICKNESS` | `bottom_absorber_thickness_um` | µm |
| `TOP ABSORBER BANDGAP` | `top_absorber_bandgap_ev` | eV |
| `BOTTOM ABSORBER BANDGAP` | `bottom_absorber_bandgap_ev` | eV |
| `T. A. DOPING DENSITY` | `top_absorber_doping_cm3` | cm⁻³; positive |
| `B. A. acceptor DENSITY` | `bottom_absorber_acceptor_cm3` | cm⁻³; positive |
| `Voc` | `Voc` | V |
| `Jsc` | `Jsc` | mA cm⁻² |
| `FF` | `FF` | % |
| `PCE` | `PCE` | % |

The code also accepts the **internal names** as input headers. Minor spacing changes in
unrecognized source headers are normalized automatically but **may not match** the
required schema, so verify the column names before attempting to train models.

Additional features derived by the notebook: `ETL`, `HTL`, log₁₀ donor/acceptor
densities, and a compact device-configuration label. Nominal device stack:
FTO/ETL/Cs₂TiBr₆/MASnI₃/HTL/Au, with ETL ∈ {PCBM, C60, TiO2} and HTL ∈
{GaAs, Cu2O, MoO3}. The raw structure string must contain recognizable transport
layer tokens.

Expected total: **18,225** valid simulated rows, with four target responses.
