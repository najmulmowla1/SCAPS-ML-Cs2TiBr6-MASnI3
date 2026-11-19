
# ============================================================
# *** 13. research-question answer table
# ============================================================

best_config = ranked_by_scaps_pce.iloc[0]
best_etl_htl = etl_htl_ranking.iloc[0]
best_top_thickness = top_thickness_effect.iloc[0]
best_bottom_thickness = bottom_thickness_effect.iloc[0]
best_top_bandgap = top_bandgap_effect.iloc[0]
best_bottom_bandgap = bottom_bandgap_effect.iloc[0]
best_top_doping = top_doping_effect.iloc[0]
best_bottom_acceptor = bottom_acceptor_effect.iloc[0]
strongest_perm_feature = permutation_importance_df.iloc[0]

research_question_answers = pd.DataFrame([
    {
        "Research_Question": "Which ETL/HTL combination provides the highest photovoltaic performance?",
        "Result_Based_Answer": f"The highest simulated PCE was obtained for ETL={best_config['ETL']} and HTL={best_config['HTL']} in the {best_config['configuration']} structure.",
        "Key_Evidence": f"Best SCAPS PCE={best_config['PCE']:.6f}%, Voc={best_config['Voc']:.6f} V, Jsc={best_config['Jsc']:.6f} mA/cm2, FF={best_config['FF']:.6f}%."
    },
    {
        "Research_Question": "How do top and bottom absorber thicknesses affect photovoltaic outputs?",
        "Result_Based_Answer": "The thickness effect is quantified through grouped mean and maximum values of Voc, Jsc, FF, and PCE.",
        "Key_Evidence": f"Best overall configuration uses top thickness={best_config['top_absorber_thickness_um']} um and bottom thickness={best_config['bottom_absorber_thickness_um']} um."
    },
    {
        "Research_Question": "How do the bandgaps of Cs2TiBr6 and MASnI3 influence performance?",
        "Result_Based_Answer": "The bandgap interaction table ranks all top/bottom bandgap combinations by photovoltaic output.",
        "Key_Evidence": f"Best overall configuration uses top bandgap={best_config['top_absorber_bandgap_ev']} eV and bottom bandgap={best_config['bottom_absorber_bandgap_ev']} eV."
    },
    {
        "Research_Question": "How do doping concentration and acceptor density affect performance?",
        "Result_Based_Answer": "The doping interaction table evaluates how top absorber doping and bottom absorber acceptor density affect Voc, Jsc, FF, and PCE.",
        "Key_Evidence": f"Best overall configuration uses top doping={best_config['top_absorber_doping_cm3']:.3e} cm^-3 and bottom acceptor density={best_config['bottom_absorber_acceptor_cm3']:.3e} cm^-3."
    },
    {
        "Research_Question": "Can machine-learning models accurately predict SCAPS-1D outputs?",
        "Result_Based_Answer": f"The best PCE prediction model was {best_pce_model_name}.",
        "Key_Evidence": f"PCE test RMSE={pce_model_metrics.loc[0, 'RMSE']:.6f}, MAE={pce_model_metrics.loc[0, 'MAE']:.6f}, R2={pce_model_metrics.loc[0, 'R2']:.6f}."
    },
    {
        "Research_Question": "Which machine-learning model provides the best prediction performance?",
        "Result_Based_Answer": f"For PCE-only prediction, {best_pce_model_name} ranked first based on test RMSE. For multi-output prediction, {best_multi_model_name} ranked first based on PCE RMSE.",
        "Key_Evidence": "See Model_Metrics_PCE_Only and Model_Metrics_All_Targets sheets."
    },
    {
        "Research_Question": "Which input parameters have the strongest effect on PCE?",
        "Result_Based_Answer": f"The strongest model-agnostic feature based on permutation importance was {strongest_perm_feature['Feature']}.",
        "Key_Evidence": f"Mean R2 decrease={strongest_perm_feature['Importance_Mean_R2_Decrease']:.6f}. See Feature_Importance_Permutation and SHAP_Global_PCE sheets."
    },
    {
        "Research_Question": "Can ML reduce the number of SCAPS-1D simulations required for optimization?",
        "Result_Based_Answer": "The trained model can rapidly screen all candidate configurations and rank high-performance structures without repeating every SCAPS-1D simulation.",
        "Key_Evidence": "See Ranked_By_ML_Predicted_PCE and Consensus_Top_100 sheets."
    },
    {
        "Research_Question": "How does the optimized dual-absorber cell compare with relevant reported devices?",
        "Result_Based_Answer": "The notebook provides the optimized configuration and photovoltaic metrics needed for comparison with reported single-absorber and dual-absorber devices.",
        "Key_Evidence": "Add literature values manually in the manuscript comparison table after final source verification."
    },
])

display(research_question_answers)
