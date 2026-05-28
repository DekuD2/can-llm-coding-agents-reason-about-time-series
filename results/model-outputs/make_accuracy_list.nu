#!/usr/bin/env nu

# You need to source it so that you can do
# main | to json | save ...
def main [
  dir: string = "."
] {
  let files = (ls $dir | where ($it.name | str ends-with ".jsonl") | get name)
  # print $files
  let accuracies = $files | each {
    |file| ./qa_accuracy.py $file --json | from json
  }
  return $accuracies
}

# Complete set of commands to use:
# ```nu
# source make_accuracy_list.nu
# let qalist = main
# $qalist | to json | save "results.json"
# $qalist | to csv | save "results.csv"
# ```
