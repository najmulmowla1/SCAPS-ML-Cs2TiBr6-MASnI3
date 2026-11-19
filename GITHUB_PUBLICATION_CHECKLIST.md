# Before publishing to GitHub

1. Create a public repository (suggested name: `SCAPS-ML-Cs2TiBr6-MASnI3`).
2. Extract the GitHub release ZIP and upload the **contents of the extracted folder**, not only the ZIP.
3. Obtain the original 18,225-configuration dataset from https://doi.org/10.17632/4fgnm4wftb.1, confirm its actual column names, and execute a complete run.
4. Verify the resulting best configuration, photovoltaic outputs, ML metrics and figures against the accepted manuscript. Document any environment/version differences.
5. Add code for the *separate* manuscript validation experiments and SCAPS-1D sensitivity sweeps if you intend to claim complete computational reproducibility. They were not in the uploaded notebook.
6. Select a repository code license; none has been assumed on behalf of the authors.
7. Once the repository is public, add its *actual URL* to the journal's code-availability statement. Do not use a placeholder URL in the proof.

**Suggested statement after verification and publication:**

> The SCAPS-1D simulation dataset is available through Mendeley Data (https://doi.org/10.17632/4fgnm4wftb.1). The Python scripts used for the available machine-learning analyses are available at [insert public GitHub repository URL].

The statement should not claim that GitHub includes the simulator, its configuration files, or the independently run sensitivity/validation experiments unless those files are subsequently added.
