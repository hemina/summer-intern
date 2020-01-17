import sys,csv
for row in csv.reader("bench_labeled_review.csv"):
  print "\t".join(row)
