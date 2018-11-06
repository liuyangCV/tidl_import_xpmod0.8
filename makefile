#******************************************************************************
#*      Copyright (C) 2009-2012 Texas Instruments Incorporated.               *
#*                      All Rights Reserved                                   *
#******************************************************************************

##############################################################
ALGBASE_PATH ?= $(abspath ../../../../)

COMMON_DIR= ../../../../common

SUBDIRS= ../tfImport/proto_cc/tensorflow/core/framework

CFILES := $(foreach dir,$(SUBDIRS),$(wildcard $(dir)/*.cc))
CFILES += ../caffeImport/caffe.pb.cc
CFILES += $(COMMON_DIR)/configparser.c

CFILES += tidl_import_main.cpp
CFILES += tidl_import_config.cpp
CFILES += tidl_import_common.cpp

#CFILES += ../tfImport/tidl_tfImport.cpp
#CFILES += ../caffeImport/tidl_caffeImport.cpp
CFILES += tidl_tfImport.cpp
CFILES += tidl_caffeImport.cpp



HFILES = ../caffeImport/caffe.pb.h
HFILES += ti_dl.h

TARGET_PLATFORM = PC

ifeq ($(TARGET_BUILD), release)
PROTOBUF_LIB_DIR?="D:\protobuf-3.2.0rc2\vs2013\Release"
else
PROTOBUF_LIB_DIR?="D:\protobuf-3.2.0rc2\vs2013\Debug"
endif
PROTOBUF_INC_DIR?="D:\protobuf-3.2.0rc2\src"



OUTFILE= tidl_model_import_noinfer.out
##############################################################

include $(ALGBASE_PATH)/makerules/rules.mk

#used inside makerules, but okay to define it afterwards
CFLAGS+= -I $(PROTOBUF_INC_DIR)
CFLAGS+= -I ..\..\inc
CFLAGS+= -I ../caffeImport
CFLAGS+= -I ../tfImport
CFLAGS+= -I ../tfImport/proto_cc
CFLAGS+= -I $(COMMON_DIR)

CFLAGS+= /D_MSC_VER=1700 /Gd
CFLAGS+= /DTIDL_IMPORT_TOOL

ifeq ($(TARGET_BUILD), debug)
CFLAGS+= /D_DEBUG /MTd
else
CFLAGS+= /D_NDEBUG /MT
endif

##############################################################

##############################################################
ifeq ($(TARGET_BUILD), release)
LDFLAGS+=-l $(PROTOBUF_LIB_DIR)/libprotoc.lib
LDFLAGS+=-l $(PROTOBUF_LIB_DIR)/libprotobuf.lib
else
LDFLAGS+=-l $(PROTOBUF_LIB_DIR)/libprotocd.lib
LDFLAGS+=-l $(PROTOBUF_LIB_DIR)/libprotobufd.lib
endif

##############################################################


##############################################################
OUTDIR =  "out"

$(OUTFILE) : outfile
##############################################################
