GC=g++
OPS=-lm -lpthread

build:  word2vec
	mv word2vec ../../../bin/word2vec
word2vec:
	$(GC) -o word2vec word2vec.c $(OPS)

clean:
	rm word2vec
