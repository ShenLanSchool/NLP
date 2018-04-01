#!/usr/bin/env bash
##
# Author: Zheng Yuanchun
# Date  : 2017-04-14 14:00:00
# Note  : This is an Linux automate shell script.
#         Running different word embeddings models with various parameters.
#         And evaluating them by compare algorithms [such as ws353]
#
# 1. call the makefile in the code source folder.
# 2. running all the word2vec application by terminal.
# 3. compare the vec with compare code.
##
CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`

echo "System: $(uname).  TimeStamp: "$CUR_DATE
echo "Running word embeddings models and algorithms, included:"
echo "  1.  cw"
echo "  2.  fastText"
echo "  3.  glove"
echo "  4.  lbl"
echo "  5.  nnlm"
echo "  6.  word2vec"

##
# 预定义一些路径，包括但是不限于：
# 1. 可运行程序的路径（bin目录的路径）
# 2. 语料库的路径
# 3. 程序的输出路径（盛放词向量的目录）
##

OS="$(uname)"
LOCAL="Darwin"      #用于判断是本机测试环境，还是服务器的运行环境

CUR_DATE=`date '+%Y%m%d_%H%M'`    #获取当前时间，为了后面统计程序的运行时间
VECSIZE=( 50 300 )                    #词向量的维度，可以添加自己要训练的维度
ITER=1                            #epoch的次数
THREADS=1                         #并行的线程数目(双核4线程最高能开到4，在服务器上用htop看有多少个线程)

# 分为在本地或是在服务器上运行。
# 考虑到本地的资源有限，所以程序运行的时候设置的参数比较简单。
# 服务器性能配置比较高，所以就在服务器上运行的时候参数就比较复杂。
if [ $OS"X" = $LOCAL"X" ]
  then
  SRC_BIN='./bin/'              #可执行文件的存放地址
  SRC_CORPUS='./corpus/'        #语料库的存放地址
  SRC_OUTPUT='./output/'        #词向量的输出地址
  ITER=1                        #epoch的次数
  THREADS=4                     #本地线程数目
  CORPUS=( "text8" )            #用于训练的不同语料
else
  SRC_BIN='./bin/'
  SRC_CORPUS='/home/zhengyuanchun/data/corpus/'    #一般语料库在服务器上都是单独放置
  SRC_OUTPUT='./output/'
  ITER=1                        #在服务器上epochs可以高点
  THREADS=30
  CORPUS=( "text8" "travel" )            #在服务器上可以添加自己收集的语料
fi

####
# 配置好了通用的运行参数之后，就可以使用不同的词向量的运行软件
# 然后再分别设置格子的参数，这样就能自动训练词向量了
####


################################################
# 训练cw词向量的单独配置
################################################
SRC_CUR_OUTPUT=${SRC_OUTPUT}"/cw/"
mkdir -p ${SRC_CUR_OUTPUT}
SRC_CUR_BIN=${SRC_BIN}"cw/cw"
CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
echo -e ${CUR_DATE}" Running \033[41;36;1m [word2vec] \033[0m model:"
for _CORPUS in ${CORPUS[@]}; do
  echo ${_CORPUS}
  _SRC_CORPUS=$SRC_CORPUS${_CORPUS}
  for _VECSIZE in ${VECSIZE[@]}; do
    cmd="(time ${SRC_CUR_BIN} -train ${_SRC_CORPUS} -cbow 1 -debug 0\
                    -hs 1 -negative 0 -iter ${ITER} -window 5\
                    -size ${_VECSIZE} -threads ${THREADS} -sample 1e-4\
                    -binary 0 -save-vocab ${SRC_CUR_OUTPUT}${_CORPUS}_vocab.txt \
                    -output "${SRC_CUR_OUTPUT}"word2vec_"${_CORPUS}"_${_MODEL}_hs_size"${_VECSIZE}".vec)"
    echo ${cmd}
    eval ${cmd}
  done
done

################################################
# 训练word2vec的词向量的单独配置
################################################
SRC_CUR_OUTPUT=${SRC_OUTPUT}"/word2vec/"          #盛放word2vec词向量的目录
mkdir -p ${SRC_CUR_OUTPUT}
SRC_CUR_BIN=${SRC_BIN}"word2vec/word2vec"         #可执行程序的名称是word2vec，已经make好了
CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
echo -e ${CUR_DATE}" Running \033[41;36;1m [word2vec] \033[0m model:"
MODEL=( "cbow" "skipgram"  )                      #word2vec可供选择的模型[CBOW Skip-gram]
for _CORPUS in ${CORPUS[@]}; do
    echo ${_CORPUS}
    _SRC_CORPUS=$SRC_CORPUS${_CORPUS}
    for _VECSIZE in ${VECSIZE[@]}; do
        for _MODEL in ${MODEL[@]}; do
            if [ $_MODEL = "cbow" ]
                then
                CUR_MODEL=1
            else
                CUR_MODEL=0
            fi
            # 同一个model分别对应不同的优化方式，一种是hierarchical softmax，一种是negative sampling
            # 下面的两个cmd分别对应着两种不同方式
            cmd="(time ${SRC_CUR_BIN} -train ${_SRC_CORPUS} -cbow ${CUR_MODEL} -debug 0\
                            -hs 1 -negative 0 -iter ${ITER} -window 5\
                            -size ${_VECSIZE} -threads ${THREADS} -sample 1e-4\
                            -binary 0 -save-vocab ${SRC_CUR_OUTPUT}${_CORPUS}_vocab.txt \
                            -output "${SRC_CUR_OUTPUT}"word2vec_"${_CORPUS}"_${_MODEL}_hs_size"${_VECSIZE}".vec)"
            echo ${cmd}
            eval ${cmd}

            cmd="(time ${SRC_CUR_BIN} -train ${_SRC_CORPUS} -cbow ${CUR_MODEL} -debug 0\
                            -hs 0 -negative 5 -iter ${ITER} -window 5\
                            -size ${_VECSIZE} -threads ${THREADS} -sample 1e-4\
                            -binary 0 -save-vocab ${SRC_CUR_OUTPUT}${_CORPUS}_vocab.txt \
                            -output "${SRC_CUR_OUTPUT}"word2vec_"${_CORPUS}"_${_MODEL}_ns_size"${_VECSIZE}".vec)"
            echo ${cmd}
            eval ${cmd}
        done
    done
done


################################################
# 训练glove的词向量的单独配置(原始demo配置请看src/embeddings/glove中的demo.sh)
################################################



##
# 1. fastText
##
SRC_CUR_OUTPUT=${SRC_OUTPUT}"/fastText/"
mkdir -p ${SRC_CUR_OUTPUT}
SRC_CUR_BIN=${SRC_BIN}"fastText/fasttext"
CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
echo -e ${CUR_DATE}" Running \033[41;36;1m [fastText] \033[0m model:"
MODEL=( "cbow" "skipgram" )
for _CORPUS in ${CORPUS[@]}; do
    echo ${_CORPUS}
    _SRC_CORPUS=${SRC_CORPUS}${_CORPUS}
    for _VECSIZE in ${VECSIZE[@]}; do
        for _MODEL in ${MODEL[@]}; do
            ## fastText 有3种不同的训练方式
            # 01.hieracial softmax
            cmd="(time ${SRC_CUR_BIN} ${_MODEL} -input ${_SRC_CORPUS} -lr 0.025 -dim ${_VECSIZE}  \
                            -ws 5 -epoch ${ITER} -minCount 5  -loss hs -neg 0 -thread ${THREADS} -t 1e-4 -verbose 0 \
                            -output "${SRC_CUR_OUTPUT}"fastText_"${_CORPUS}"_${_MODEL}_hs_size"${_VECSIZE}")"
            echo ${cmd}
            eval ${cmd}
            # 02.negative sampling
            cmd="(time ${SRC_CUR_BIN} ${_MODEL} -input ${_SRC_CORPUS} -lr 0.025 -dim ${_VECSIZE}  \
                            -ws 5 -epoch ${ITER} -minCount 5  -loss ns -neg 5 -thread ${THREADS} -t 1e-4 -verbose 0 \
                            -output "${SRC_CUR_OUTPUT}"fastText_"${_CORPUS}"_${_MODEL}_ns_size"${_VECSIZE}")"
            echo ${cmd}
            eval ${cmd}
            # 03.softmax

        done
    done
done


# SRC_CUR_OUTPUT=${SRC_OUTPUT}"nnlm"
# mkdir ${SRC_CUR_OUTPUT}
# SRC_CUR_BIN=${SRC_BIN}"nnlm"
# CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
# echo ${CUR_DATE}" Running [nnlm] model"

# SRC_CUR_OUTPUT=${SRC_OUTPUT}"cw"
# mkdir ${SRC_CUR_OUTPUT}
# SRC_CUR_BIN=${SRC_BIN}"cw"
# CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
# echo ${CUR_DATE}" Running [cw] model"

# SRC_CUR_OUTPUT=${SRC_OUTPUT}"lbl"
# mkdir ${SRC_CUR_OUTPUT}
# SRC_CUR_BIN=${SRC_BIN}"lbl"
# CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
# echo ${CUR_DATE}" Running [lbl] model"

# SRC_CUR_OUTPUT=${SRC_OUTPUT}"order"
# mkdir ${SRC_CUR_OUTPUT}
# SRC_CUR_BIN=${SRC_BIN}"order"
# CUR_DATE=`date '+%Y:%m:%d %H:%M:%S'`
# echo ${CUR_DATE}" Running [order] model"
