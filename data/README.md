# Data access

This repository does not contain ICILS microdata.

1. Open the dataset page via its DOI (https://doi.org/10.58150/ICILS_2023_data_2) and follow IEA's registration and download steps.
2. Download the **ICILS 2023 International Database, Edition 2**, SPSS format.
3. The analysis needs only the student files `BSG*.sav` (one per education system). You can upload the whole zip; the notebook extracts only `BSG*.sav`, including from nested zips.

Edition 2 is required: it added the Farm Drone process variables used here (task times `T2F01_T` … `T2F09_T` and resets `T2F01BA_P` … `T2F08BA_P`; tasks 4 and 6 use the prefix `T3`).

Variables used (beyond the CT item variables listed in `results/01_variable_map.csv`): `IDCNTRY`, `CNTRY`, `IDSCHOOL`, `IDSTUD`, `TOTWGTS`, `JKZONES`, `JKREPS`, `PV1CT`–`PV5CT` (sample selection only), `PV1CIL`–`PV5CIL`, `S_GENDER` (or `S_SEX`), and `IISB`.
