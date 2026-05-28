# Outputs

This directory contains the outputs of the models (.jsonl files packed inside .tar.gz) answersing the TSE and TSFU dataset questions.

`accuracy-table.csv` contains the summary of all joint and categorical accuracies across models and setups.

Use `./accuracy.py model-outputs--tse-gpt-oss-hybrid.jsonl` to view the individual accuracies.

The nushell script `./make_accuracy_list.nu` can be used to create the joint table of accuracies. See info inside for usage.
