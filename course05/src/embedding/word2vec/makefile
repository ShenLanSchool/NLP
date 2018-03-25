GC=g++
OPS=-lm -lpthread

build:  word2vec word2phrase distance word-analogy compute-accuracy
	mv word2vec ../../../bin/word2vec
	mv word2phrase distance word-analogy compute-accuracy ../../evaluation

word2vec: word2vec.c
	$(GC) -o word2vec word2vec.c $(OPS)

word2phrase: word2phrase.c
	$(GC) word2phrase.c -o word2phrase $(OPS)

distance: distance.c
	$(GC) distance.c -o distance $(OPS)

word-analogy: word-analogy.c
	$(GC) word-analogy.c word-analogy $(OPS)

compute-accuracy: compute-accuracy.c
	$(GC) compute-accuracy.c -o compute-accuracy $(OPS)

clean:
	rm word2vec word2phrase distance word-analogy compute-accuracy
