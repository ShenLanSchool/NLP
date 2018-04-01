#!/bin/sh

mvn compile -X \
  | grep 'classpathElements = ' \
  | sed 's#^.* classpathElements = \[\(.*\)\]$#\1#g' \
  | sed 's#, #:#g' \
  &> CP.hack
export MY_CP_PREPROCESS=`cat CP.hack`

