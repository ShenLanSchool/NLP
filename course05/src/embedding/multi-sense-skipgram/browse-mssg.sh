#!/bin/sh
## scipt for computing the vectors  and checking the accuracy with google's questions-words analogy task

if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit 
fi


classpath=`cat CP.hack`
nearest_neighbour_app="java -Xmx100g -cp ${classpath} MultiSenseEmbeddingBrowse"

${nearest_neighbour_app} --embedding vectors-MSSG.gz --num-neighbour 20
