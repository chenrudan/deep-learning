#################################                                                                                                                                                 
#  Compiler, tools and options  #
#################################

CXX = g++ 
DEBUG = -g
CFLAGES = -Wall -c $(DEBUG)
LFLAGES = -Wall $(DEBUG)
LINK = g++ 
CXXFLAGES = -O3 -DADD_
MC = mpicc
MCXX=mpicxx
#CBLASPATH = /home/crd/crd/deeplearning/CBLAS/CBLAS/
#BLASPATH = /home/crd/crd/deeplearning/CBLAS/BLAS/
BLASPATH = /usr/local/lib
CBLASPATH = /home/crd/crd/github/deep/cblas/CBLAS/include/
INCLUDE = $(CBLASPATH)
CBLIB =  ./lib/libcblas.a ./lib/libblas.a
####### Compile


showthree: showthree.cpp
	$(CXX) $(CFLAGES) showthree.cpp
	$(LINK) -o showthree showthree.o `pkg-config --libs opencv`

showweight: showweight.cpp
	$(CXX) $(CFLAGES) showweight.cpp
	$(LINK) -o showweight showweight.o `pkg-config --libs opencv`

show: show.cpp
	$(CXX) $(CFLAGES) show.cpp
	$(LINK) -o show show.o `pkg-config --libs opencv`

svm: svmclassify.cpp
	$(CXX) $(CFLAGES) svmclassify.cpp
	$(LINK) -o svm svmclassify.o `pkg-config --libs opencv`

get_feature: get_feature.cpp
	$(CXX) $(CFLAGES) get_feature.cpp -I $(INCLUDE)
	$(LINK) -o get_feature get_feature.o  $(CBLIB) `pkg-config --libs opencv`  -lgfortran

showlayer2: showlayer2.cpp
	$(CXX) $(CFLAGES) showlayer2.cpp -I $(INCLUDE)
	$(LINK) -o showlayer2 showlayer2.o  $(CBLIB) `pkg-config --libs opencv`  -lgfortran

savefeature: savefeature.cpp
	$(CXX) $(CFLAGES) savefeature.cpp -I $(INCLUDE)
	$(LINK) -o savefeature savefeature.o  $(CBLIB) `pkg-config --libs opencv`  -lgfortran

savel1: savel1.cpp
	$(CXX) $(CFLAGES) savel1.cpp -I $(INCLUDE)
	$(LINK) -o savel1 savel1.o  $(CBLIB) `pkg-config --libs opencv`  -lgfortran

cleanall:
	rm -f *.o show svmclassify showthree showweight get_feature showlayer2 savefeature savel1
