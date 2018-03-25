MAKEPATH=./src/embedding/
build:
	mkdir -p output bin
	echo -e "\033[41;36m compling c&w \033[0m"
	mkdir -p bin/cw
	make -C $(MAKEPATH)cw

	echo -e "\033[41;36m compling fastText \033[0m"
	mkdir -p bin/fastText
	make -C $(MAKEPATH)fastText

	echo -e "\033[41;36m compling glove \033[0m"
	mkdir -p bin/glove
	make -C $(MAKEPATH)glove

	echo -e "\033[41;36m lbl \033[0m"
	mkdir -p bin/lbl
	make -C $(MAKEPATH)lbl

	echo -e "\033[41;36m compling nnlm \033[0m"
	mkdir -p bin/nnlm
	make -C $(MAKEPATH)nnlm

	echo -e "\033[41;36m compling word2vec \033[0m"
	mkdir -p bin/word2vec
	make -C $(MAKEPATH)word2vec

clean:
	rm -rf bin/
