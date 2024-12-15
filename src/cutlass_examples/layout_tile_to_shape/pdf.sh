#!/bin/bash

set -e

for tex in $(ls *.tex)
do 
  pdflatex ${tex} > latex.log 2>&1
done

code *.pdf
