#!/bin/sh
if [ ! -e CP.hack ];  then
    echo 'Run make_cp.sh first!'
    `make_cp.sh`
fi


datapath="/home/public/Documents/zhengyuanchun/corpus/wiki/"
outpath="/home/public/Documents/zhengyuanchun/output/"
corpus=( "wiki8.txt" "wiki9.txt" "wiki7.txt" )
vecsize=( 50 100 )

currentOutput=${outpath}"10_mssg/"
mkdir ${currentOutput}
for _corpus in ${corpus}; do
  for _vecsize in ${vecsize}; do

      curCorpus=${datapath}${_corpus}
      curOutput=${currentOutput}${_corpus}${_vecsize}.mssg.vec

      classpath=`cat CP.hack`
      wordvec_app="java -Xmx100g -cp ${classpath} WordVec"
      cmd="${wordvec_app}  --model=MSSG-MaxOut \
                    --train=${curCorpus}  \
                    --output=${curOutput}    \
                    --sense=3    \
                    --learn-top-v=4000   \
                    --size=${_vecsize}   \
                    --window=5   \
                    --min-count=5   \
                    --threads=30   \
                    --negative=1   \
                    --sample=0.001   \
                    --binary=0   \
                    --ignore-stopwords=1   \
                    --encoding=ISO-8859-15   \
                    --save-vocab=${_corpus}.vocab   \
                    --rate=0.025 --delta=0.1"
      eval ${cmd}
  done
done
