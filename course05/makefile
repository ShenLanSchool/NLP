#call all the sub-folder makefile

MAKEPATH=./src/embedding/
build:
	mkdir -p output bin

	echo 'compling word2vec'
	mkdir -p bin/word2vec
	make -C $(MAKEPATH)word2vec

	echo 'compling fastText'
	mkdir bin/fastText
	make -C $(MAKEPATH)fastText

	echo 'compling Glove'
	mkdir bin/glove
	make -C $(MAKEPATH)glove

clean:
	cd bin
