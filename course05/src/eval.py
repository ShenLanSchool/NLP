#coding:utf-8

# Name: Zhengyuanchun
# Date: 2018-03-27
# Anno: 词向量的内部评价标准和指标计算。

import gensim
import sys
import numpy as np

def cosDistance(v1,v2):
	return

class Color(object):

	__COLOR__={"red":("\033[1;31;40m","\033[0m"),
	           "green":("","")}
	@classmethod
	def color(self,text,color="red"):
		if color not in self.__COLOR__.keys():
			print("[%s] \033[1;31;40m unsupported!\033[0m")
			color="red"
		return self.__COLOR__[color][0]+str(text)+self.__COLOR__[color][1]

class Eval(object):

	def __init__(self):
		self.anayFiles = [
			'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
			'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
			'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
			'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
			'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
		]
		self.anayPathPrefix = '../datasets/question-data/'

	def loadVectors(self,vectorFile,vocabFile=None):
		"""
		加载词向量，词向量现在只支持text文件，并且第一行必须是词向量个数，维度
		:param vectorFile: vector file
		:param vocabFile: vocabulary file (default is None)
		:return:
		"""
		self.vectorFile=vectorFile
		self.vocabFile=vocabFile
		print("正在加载 %s"%(Color.color(self.vectorFile)))
		# todo: 这里修改了glove的源码，使得输出的二进制文件也有第一行的信息
		self.model=gensim.models.KeyedVectors.load_word2vec_format(self.vectorFile,fvocab=self.vocabFile)
		print("词向量数目：%s \n词向量维度：%s "%(Color.color(self.model.vocab.__sizeof__()),Color.color(self.model.vector_size)))
		return self

	def distance(self,topN=20):
		targetWord=raw_input("请输入一个单词(Exit 退出):")
		while targetWord != "Exit":
			for result in self.model.most_similar(targetWord,topn=topN):
				print("%10s : %7.5f"%(result))

			targetWord=raw_input("继续输入一个单词(Exit 退出):")
		return self

	def anaySingle(self,fileName):
		"""
		单个的类比测试文件的统计
		:param fileName:  文件名
		:return:(文件名，总样例数目，正确样例数目，能找到的总数目)
		"""
		nbTotal,nbRight,nbInclude=0,0,0
		with open(fileName) as f:
			lines=f.readlines()
			nbTotal=len(lines)
			for [w1, w2 ,w3,w4] in [line.strip().split(" ") for line in lines]:
				if  w1 in self.model.vocab and \
					w2 in self.model.vocab and \
					w3 in self.model.vocab:
					_w4Vector=self.model[w3]-self.model[w1]+self.model[w2]
					[(_w4,_)] = self.model.similar_by_vector(_w4Vector,topn=1)
					nbInclude+=1
					if _w4 == w4:
						nbRight+=1
		return (fileName,nbTotal,nbRight,nbInclude)

	def anay(self,anayFile=None):
		nbTotal,nbRight,nbInclude=0,0,0

		for anayFile in self.anayFiles[0:8]:
			anayFilePath=self.anayPathPrefix+anayFile
			_file,_nbTotal,_nbRight,_nbInclude=self.anaySingle(anayFilePath)
			nbTotal+=_nbTotal
			nbInclude+= _nbInclude
			nbRight+= _nbRight
			print("%s:\nTotal:%4d \nRight:%4d \nInclude:%4d\n %3.2f %3.2f"%\
			      (_file,_nbTotal,_nbRight,_nbInclude,float(_nbRight)/_nbTotal,float(_nbRight)/_nbInclude))

		print("Summary: Total:%4d Right:%4d Include:%4d\t %3.2f %3.2f" % \
		      ( _nbTotal, _nbRight, _nbInclude, float(_nbRight) / _nbTotal, float(_nbRight) / _nbInclude))
		return self

	def sim(self,wsFile=None):


if __name__ == "__main__":
	eval=Eval()
	eval.loadVectors("./embedding/glove/vectors.txt").sim()
