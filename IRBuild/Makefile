ASCEND_PATH := /usr/local/Ascend

LOCAL_DIR  := ./
ATC_INCLUDE_DIR := $(ASCEND_PATH)/atc/include
OPP_INCLUDE_DIR := $(ASCEND_PATH)/opp/op_proto/built-in/inc

LOCAL_MODULE_NAME := ir_build
CC := g++
CFLAGS := -std=c++11 -g -Wall -D_GLIBCXX_USE_CXX11_ABI=0
SRCS := $(wildcard $(LOCAL_DIR)/main.cpp)

INCLUDES := -I $(ASCEND_PATH)/opp/op_proto/built-in/inc \
            -I $(ATC_INCLUDE_DIR)/graph \
            -I $(ATC_INCLUDE_DIR)/ge \
            -I $(ATC_INCLUDE_DIR)/parser \
            -I $(ASCEND_PATH)/atc/include \

LIBS := -L ${ASCEND_PATH}/atc/lib64/stub \
    -lgraph \
    -lge_compiler \
    -lfmk_parser \

ir_build:
	mkdir -p out
	$(CC) $(SRCS) $(INCLUDES) $(LIBS) $(CFLAGS) -o ./out/$(LOCAL_MODULE_NAME)
clean:
	rm -rf out
