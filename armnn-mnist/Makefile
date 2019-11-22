# ARMNN_LIB = ${HOME}/devenv/build-x86_64/release/armnn
# ARMNN_INC = ${HOME}/devenv/armnn/include

ARMNN_LIB = /home/liurenju/Desktop/research/mobisys-2020/ML-examples/armnn-mnist
ARMNN_INC = /home/liurenju/Desktop/research/mobisys-2020/ML-examples/armnn-mnist/armnn_include
ARMNN_INC_BOOST = /home/liurenju/Desktop/research/mobisys-2020/ML-examples/armnn-mnist/boost_include

all: mnist_caffe mnist_tf

mnist_caffe: mnist_caffe.cpp mnist_loader.hpp
	aarch64-linux-gnu-g++ -O3 -std=c++14 -I$(ARMNN_INC) -I$(ARMNN_INC_BOOST) mnist_caffe.cpp -o mnist_caffe -L$(ARMNN_LIB) -larmnn -larmnnCaffeParser -lprotobuf -larmnnUtils

mnist_tf: mnist_tf.cpp mnist_loader.hpp
	aarch64-linux-gnu-g++ -O3 -std=c++14 -I$(ARMNN_INC) -I$(ARMNN_INC_BOOST) mnist_tf.cpp -o mnist_tf -L$(ARMNN_LIB) -larmnn -larmnnTfParser -lpthread -lprotobuf -larmnnUtils

clean:
	-rm -f mnist_tf mnist_caffe

test: mnist_caffe mnist_tf
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_tf
