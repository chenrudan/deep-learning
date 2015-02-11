#################################                                                                                                                                           
#  Compiler, tools and options  #
#################################

CXX = g++ 
DEBUG = -g
CFLAGES = -Wall -c $(DEBUG)
LFLAGES = -Wall $(DEBUG)
LINK = g++ 
CXXFLAGES = -O3 -DADD_

####### Compile

test: 
	$(CXX) $(CFLAGES) test.cpp
	$(CXX) $(CFLAGES) matrix.cpp preprocess.cpp conv.cpp crbm.cpp load.cpp show.cpp `pkg-config --cflags opencv` 
	$(LINK) -o test test.o matrix.o preprocess.o conv.o crbm.o load.o show.o `pkg-config --libs opencv` 
	

clearall:
	rm -f *.o test
