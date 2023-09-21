#==============================================================================
# Paths
#==============================================================================
DEMO_DIR = .

BIN_DIR = $(DEMO_DIR)/bin
EXT_DIR = $(DEMO_DIR)/externals
SRC_DIR = $(DEMO_DIR)/src
LIB_DIR = $(DEMO_DIR)/lib
INCLUDE_DIR = $(DEMO_DIR)/include
OBJ_DIR = $(DEMO_DIR)/obj
PYTHON3_DIR = $(DEMO_DIR)/python3
BUILD_DIR = $(DEMO_DIR)/build

CC = gcc -g
CXX = g++ -g
CFLAGS = -Wall -fPIC -std=gnu11 -pedantic -Wno-unused-result -O3 -fopenmp
DEMOFLAGS = -Wall -fPIC -pedantic -Wno-unused-result -O3 -fopenmp
CXXFLAGS = -O3 -lstdc++ -fPIC -ffast-math -march=skylake -mfma -Wall -Wno-unused-result
LIBS = -lm

HEADER_INC = -I $(EXT_DIR) -I $(INCLUDE_DIR)
LIB_INC = -L $(LIB_DIR) -lidisf


#==============================================================================
# Rules
#==============================================================================
.PHONY: createdir all c clean lib

all: createdir lib c

lib: obj
	$(eval ALL_OBJS := $(wildcard $(OBJ_DIR)/*.o))
	ar csr $(LIB_DIR)/libidisf.a $(ALL_OBJS)


lib2: obj
	$(eval ALL_OBJS := $(wildcard $(OBJ_DIR)/*.o))

obj: \
	$(OBJ_DIR)/Utils.o \
	$(OBJ_DIR)/IntList.o \
	$(OBJ_DIR)/Color.o \
	$(OBJ_DIR)/PrioQueue.o \
	$(OBJ_DIR)/Image.o \
	$(OBJ_DIR)/Graph.o \
	$(OBJ_DIR)/iDISF.o 

createdir:
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi
	if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi
	if [ ! -d $(LIB_DIR) ]; then mkdir -p $(LIB_DIR); fi


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDE_DIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@ $(HEADER_INC) $(LIBS)

c: lib
	$(CXX) $(DEMOFLAGS) $(CXXFLAGS) iDISF_demo.c -o $(BIN_DIR)/iDISF_demo $(HEADER_INC) $(LIB_INC) $(LIBS)

python3: lib
	python3 python3/setup.py install --user --record $(PYTHON3_DIR)/dir_libs.txt
	python3 python3/setup.py clean;

clean:
	if [ -d $(OBJ_DIR) ]; then rm -r $(OBJ_DIR); fi
	if [ -d $(BIN_DIR) ]; then rm -r $(BIN_DIR); fi
	if [ -d $(BUILD_DIR) ]; then rm -r $(BUILD_DIR); fi
	if [ -d $(LIB_DIR) ]; then rm -r $(LIB_DIR); fi
	rm -rf $(PYTHON3_DIR)/*.so ;
	if [ -f "$(PYTHON3_DIR)/dir_libs.txt" ]; then xargs rm -rf < $(PYTHON3_DIR)/dir_libs.txt; rm -r $(PYTHON3_DIR)/dir_libs.txt; fi
	if [ -f "$(DEMO_DIR)/opencv_sed_model.yml.gz" ]; then rm -f $(DEMO_DIR)/opencv_sed_model.yml.gz; fi

