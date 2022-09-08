# MARGINAL-software

# What it does? 

MARGINAL is a software that includes a machine learning model to support variant classification in BRCA1 and BRCA2 genes based on ACMG-AMP guidelines.

# Excel files

-  clinvar_roi_annotation.xlsx: clinvar annotation
- Fisher_exact_test_FINAL_patients_AC.xlsx: Case-study dataset
- output_vep_clinvar_missense_score_filtered.xlsx: missense variants derived from ClinVar and annotated by VEP
- repeatmasker_brca1_2.xlsx: repeat regions in BRCA1/2 genes
- final annotation file_CanVaS.xlsx: annotation file extracted based on CanVaS database
- final annotation file_ClinVar.xlsx: annotation file extracted based on ClinVar database
- Final_features_labels_CanVaS.xlsx: extracted features with labels based on CanVaS database for training and testing machine learning model
- Final_features_labels_ClinVar.xlsx: extracted features with labels based on ClinVar database for testing machine learning model (verification)


# Script files

- MARGINAL_software_annotation.py: Annotation and feature extraction by MARGINAL software
- machine_learning_model.py: This script creates the final machine learning model for variant classification
