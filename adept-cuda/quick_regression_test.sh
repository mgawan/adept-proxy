#!/bin/bash
srun ./program_gpu dna ../../test-data/dna-reference.fasta     ../../test-data/dna-query.fasta     test_regression_dna_out     >/dev/null 2>&1
srun ./program_gpu aa  ../../test-data/protein-reference.fasta ../../test-data/protein-query.fasta test_regression_protein_out >/dev/null 2>&1

diff_dna=$(diff -b test_regression_dna_out     ../../test-data/results_dna)
diff_aa=$(diff -b test_regression_protein_out ../../test-data/results_aa)

if [ "$diff_dna" == "" ] && [ "$diff_aa" == "" ]
then
  echo "REGRESSION TESTS PASS"
  exit 0
else
  echo "#############DNA#############"
  echo "$diff_dna"
  echo "#############AA#############"
  echo "$diff_aa"
  exit 1
fi
