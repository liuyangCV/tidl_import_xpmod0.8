/*
*
* Copyright (c) {2015 - 2017} Texas Instruments Incorporated
*
* All rights reserved not granted herein.
*
* Limited License.
*
* Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
* license under copyrights and patents it now or hereafter owns or controls to make,
* have made, use, import, offer to sell and sell ("Utilize") this software subject to the
* terms herein.  With respect to the foregoing patent license, such license is granted
* solely to the extent that any such patent is necessary to Utilize the software alone.
* The patent license shall not apply to any combinations which include this software,
* other than combinations with devices manufactured by or for TI ("TI Devices").
* No hardware patent is licensed hereunder.
*
* Redistributions must preserve existing copyright notices and reproduce this license
* (including the above copyright notice and the disclaimer and (if applicable) source
* code license limitations below) in the documentation and/or other materials provided
* with the distribution
*
* Redistribution and use in binary form, without modification, are permitted provided
* that the following conditions are met:
*
* *       No reverse engineering, decompilation, or disassembly of this software is
* permitted with respect to any software provided in binary form.
*
* *       any redistribution and use are licensed by TI for use only with TI Devices.
*
* *       Nothing shall obligate TI to provide you with source code for the software
* licensed and provided to you in object code.
*
* If software source code is provided to you, modification and redistribution of the
* source code are permitted provided that the following conditions are met:
*
* *       any redistribution and use of the source code, including any resulting derivative
* works, are licensed by TI for use only with TI Devices.
*
* *       any redistribution and use of any object code compiled from the source code
* and any resulting derivative works, are licensed by TI for use only with TI Devices.
*
* Neither the name of Texas Instruments Incorporated nor the names of its suppliers
*
* may be used to endorse or promote products derived from this software without
* specific prior written permission.
*
* DISCLAIMER.
*
* THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <io.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "ti_dl.h"
#include "caffe.pb.h"
#include <cfloat>
#include "tidl_import_config.h"

using namespace std;
using namespace caffe;
using ::google::protobuf::Message;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::FileOutputStream;
using ::google::protobuf::io::ZeroCopyInputStream;
using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::ZeroCopyOutputStream;
using ::google::protobuf::io::CodedOutputStream;
#include "tidl_import_common.h"

int quantizationStyle;


//for output caffemodel file only
extern char *gpModeltxtFile;
extern int giopmodeltxtEn;

#define QUAN_STYLE01_ROUND (0.5)

#define ENABLE_FIXED_QUANT_STYLE (0)

int TIDL_flApply(float data, int fl)
{
  int out;
  if( data > 0)
  {
    out = (data* (1 << fl) + QUAN_STYLE01_ROUND);
  }
  else
  {
    out = (data* (1 << fl) - QUAN_STYLE01_ROUND);
  }
  return out;
}

#define IS_SIGNED_DATA (1)

int32_t TIDL_QuantizeP2(int8_t * params, float * data, int32_t dataSize, float min, float max)
{
  int32_t i;
  float absMax = abs(min) > abs(max) ? abs(min) : abs(max);
  float absMaxP2 = pow(2, ceil(log(absMax)/log(2)));

  float fl_range = ((1.0*(1 << (NUM_WHGT_BITS - IS_SIGNED_DATA))) / absMaxP2);
  int32_t fl = (int32_t)log(fl_range)/log(2);

  for(i = 0; i < dataSize; i++)
  {
    params[i] = TIDL_flApply(data[i], fl);
  }
  return fl;
}

int32_t TIDL_isInputEltWise(sTIDL_OrgNetwork_t  *pOrgTIDLNetStructure,
int32_t             numLayer,
const char          *bufName)
{
  int32_t i,j;
  for (i = (numLayer-1); i >= 0; i--)
  {
    for (j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs; j++)
    {
      if(strcmp((const char*)bufName,
            (const char*)pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[j]) == 0)
      {
        if((pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs == 1) &&
            (pOrgTIDLNetStructure->TIDLPCLayers[i].layerType == TIDL_EltWiseLayer))
        {
          return 1 ;
        }
        else
        {
          return 0 ;
        }
      }
    }
  }
  return 0;
}

int32_t TIDL_isInputInnreProduct(sTIDL_OrgNetwork_t *pOrgTIDLNetStruct,
int32_t            numLayer,
const char         *bufName)
{
  int32_t i,j;
  for (i = (numLayer-1); i >= 0; i--)
  {
    for (j = 0; j < pOrgTIDLNetStruct->TIDLPCLayers[i].numOutBufs; j++)
    {
      if(strcmp((const char*)bufName,
            (const char*)pOrgTIDLNetStruct->TIDLPCLayers[i].outDataNames[j]) == 0)
      {
        if((pOrgTIDLNetStruct->TIDLPCLayers[i].numOutBufs == 1) &&
            (pOrgTIDLNetStruct->TIDLPCLayers[i].layerType==TIDL_InnerProductLayer))
        {
          return 1 ;
        }
        else
        {
          return 0 ;
        }
      }
    }
  }
  return 0;
}
int32_t TIDL_isInputBatchNorm(sTIDL_OrgNetwork_t *pOrgTIDLNetStruct,
int32_t            numLayer,
const char         *bufName)
{
  int32_t i,j;
  for (i = (numLayer-1); i >= 0; i--)
  {
    for (j = 0; j < pOrgTIDLNetStruct->TIDLPCLayers[i].numOutBufs; j++)
    {
      if(strcmp((const char*)bufName,
            (const char*)pOrgTIDLNetStruct->TIDLPCLayers[i].outDataNames[j]) == 0)
      {
        if((pOrgTIDLNetStruct->TIDLPCLayers[i].numOutBufs == 1) &&
            (pOrgTIDLNetStruct->TIDLPCLayers[i].layerType==TIDL_BatchNormLayer))
        {
          return 1 ;
        }
        else
        {
          return 0 ;
        }
      }
    }
  }
  return 0;
}

int32_t TIDL_inputLayerIndex(sTIDL_OrgNetwork_t *pOrgTIDLNetStructure,
int32_t            numLayer,
const char         *bufName)
{
  int32_t i,j;
  for (i = (numLayer-1); i >= 0; i--)
  {
    for (j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs; j++)
    {
      if(strcmp((const char*)bufName,
            (const char*)pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[j]) == 0)
      {
        return i;
      }
    }
  }
  return -1;
}

int32_t TIDL_isBottomUsedLater(const char* botName, int32_t startLayer,
NetParameter &netStructure)
{
  int32_t i,j;
  for (i = startLayer+1; i < netStructure.layer_size(); i++)
  {
    for (j = 0; j < netStructure.layer(i).bottom_size(); j++)
    {
      if(strcmp((const char*)botName,
            (const char*)netStructure.layer(i).bottom(j).c_str()) == 0)
      {
        return true;
      }
    }
  }
  return false;
}

int TIDL_appCNNInteropCaffeFindLayerByName(const char *name,
const NetParameter &netParams)
{
  int i;
  int layerNum = netParams.layer_size();
  for (i = 0; i < layerNum; i++) {
    if (string(name) == netParams.layer(i).name())
    {
      return i;
    }
  }
  return -1;
}


int TIDL_appCNNConverRawDataToData(NetParameter &netParams)
{
  int i,j,k;
  int layerNum = netParams.layer_size();
  for (i = 0; i < layerNum; i++)
  {
    for (j = 0; j < netParams.layer(i).blobs_size(); j++)
    {
      if(netParams.layer(i).blobs(j).has_raw_data())
      {
        Type raw_type = netParams.layer(i).blobs(j).raw_data_type();
        const ::std::string& hd = netParams.layer(i).blobs(j).raw_data();
        if (raw_type == caffe::FLOAT )
        {
          int data_size = hd.size() / 4;
          float *Y = (float*)(&hd.front());
          BlobProto & blob = (BlobProto&)netParams.layer(i).blobs(j);
          for(k = 0 ; k < data_size; k++)
          {
            blob.add_data(Y[k]);
          }
        }
        else
        {
          printf("Un supported raw_dat_tyep\n");
          return -1;
        }
      }
    }
  }
  return -1;
}

bool Tidl_ExportBinaryDataToTxtData(const Message & netParams)
{
  int32_t           fd;
  bool              success;
  FileOutputStream   *output;

  if(gpModeltxtFile != NULL && gpModeltxtFile[0]!='\0')
  {
    printf("Writing caffemodel into txt file %s...",gpModeltxtFile);
    fd = open(gpModeltxtFile, O_WRONLY | O_CREAT);
  }
  else{
    printf("Writing caffemodel into txt file .\\CaffeNetWorkParameters.txt...");
    fd = open("CaffeNetWorkParameters.txt", O_WRONLY | O_CREAT);
  }
  output = new FileOutputStream(fd);
  success = google::protobuf::TextFormat::Print(netParams, output);

  delete output;
  close(fd);
  printf("complete!\n");
  return success;
}

void TIDL_importConcatParams(sTIDL_OrgNetwork_t *pOrgTIDLNetStructure,
int32_t            i,
int32_t            layerIndex,
int32_t            dataIndex,
NetParameter       netStructure)
{
  int32_t j, status, numOuChs = 0;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType      =
  TIDL_ConcatLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs     = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],
  netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId =
  dataIndex++;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs      =
  netStructure.layer(i).bottom_size();

  for(j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs; j++)
  {
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[j],
    netStructure.layer(i).bottom(j).c_str());
    status = TIDL_getDataID(
    &pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[j],
    pOrgTIDLNetStructure,
    layerIndex,
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[j]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[j]);
      exit(-1);
    }
    numOuChs +=
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[j].dimValues[1];
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] =
  numOuChs;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs = 0;
}

void TIDL_importConvParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
FILE                 *fp1,
int32_t              i,
int32_t              layerIndex,
int32_t              dataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t             status, id;
  int32_t             padH, padW, stride;
  int32_t             paramSet = 0;
  int32_t             dataSize;
  int32_t             conv2DRandParams = 0;
  int32_t             prevLayerIdx = 0;
  float min, max;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType    =
  TIDL_ConvolutionLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs   =   1;
  // printf("Layer num: %d, checkpoint (4)\n",i);
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],
  netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId =
  dataIndex++;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs    =   1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],
  netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(
  &pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0],
  pOrgTIDLNetStructure,
  layerIndex,
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numInChannels  =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numOutChannels =
  netStructure.layer(i).convolution_param().num_output();
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups      =
  netStructure.layer(i).convolution_param().group();
  // printf("Layer num: %d, checkpoint (4)\n",i);
  if(netStructure.layer(i).convolution_param().has_kernel_w()){
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW        =
    netStructure.layer(i).convolution_param().kernel_w();//kernel_size(0);
  }
  else{
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW        =
    netStructure.layer(i).convolution_param().kernel_size(0);
  }
  // printf("Layer num: %d, checkpoint (5)\n",i);
  if (netStructure.layer(i).convolution_param().has_kernel_h()) {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH        =
    netStructure.layer(i).convolution_param().kernel_h();//kernel_size(0);
  }
  else{
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH        =
    netStructure.layer(i).convolution_param().kernel_size(0);
  }
  // printf("Layer num: %d, checkpoint (6)\n",i);
  if (netStructure.layer(i).convolution_param().has_pad_h()) {
    padH =  netStructure.layer(i).convolution_param().pad_h();
  }
  else{
    if (netStructure.layer(i).convolution_param().pad_size() == 0) {
      padH = 0;
    }
    else {
      padH = netStructure.layer(i).convolution_param().pad(0);
    }
  }
  if (netStructure.layer(i).convolution_param().has_pad_w()) {
    padW =  netStructure.layer(i).convolution_param().pad_w();
  }
  else{
    if (netStructure.layer(i).convolution_param().pad_size() == 0) {
      padW = 0;
    }
    else {
      padW = netStructure.layer(i).convolution_param().pad(0);
    }
  }

  if (netStructure.layer(i).convolution_param().stride_size() == 0) {
    stride = 1;
  }
  else {
    stride = netStructure.layer(i).convolution_param().stride(0);
  }

  if (netStructure.layer(i).convolution_param().dilation_size() == 0) {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW = 1;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH = 1;
  }
  else {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW =
    netStructure.layer(i).convolution_param().dilation(0);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideW=
  stride;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideH=
  stride;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padW   =
  padW;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padH   =
  padH;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numOutChannels;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] =
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] +
  (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padH*2)-
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH-1)*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH+1))/
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideH) + 1;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] =
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3] +
  (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padW*2)-
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW-1)*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW + 1))/
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideW) + 1;

  if(gParams.conv2dKernelType == 0)
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelType = TIDL_sparse;
  }
  else if(gParams.conv2dKernelType == 1)
  {
    if((((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW == 1 ) &&
            (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH == 1 )) ||
          ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW == 3 ) &&
            (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH == 3 ))) &&
        ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideW == 1 ) &&
          (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideH == 1 )) )
    {

      if((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] < 64) ||
          (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] < 64) )
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelType = TIDL_dense;
      }
    }
  }
  // printf("Layer num: %d, checkpoint (4)\n",i);
  if(quantizationStyle == TIDL_quantStyleFixed)
  {
#if ENABLE_FIXED_QUANT_STYLE
    if(netStructure.layer(i).quantization_param().unsigned_layer_out())
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType =
      TIDL_UnsignedChar;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType =
      TIDL_SignedChar;
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.inDataQ =
    netStructure.layer(i).quantization_param().fl_layer_in(0);
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.outDataQ =
    netStructure.layer(i).quantization_param().fl_layer_out();
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weightsQ =
    netStructure.layer(i).quantization_param().fl_weights();
#endif
  }
  else
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType =
    TIDL_SignedChar;
  }

  id = TIDL_appCNNInteropCaffeFindLayerByName(
  (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  paramSet = 0;
  if((id != -1) && (conv2DRandParams == 0))
  {
    if(netParams.layer(id).blobs_size() > 0)
    {
      dataSize = netParams.layer(id).blobs(0).data_size();
      if(dataSize !=
          ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1])/
            pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups))
      {
        printf("Kernel Size not matching %d !!", dataSize);
      }
      else
      {
        paramSet = 1;
        float  * data   = (float *)malloc(dataSize*sizeof(float));
        for (int idx = 0; idx < dataSize; idx++)
        {
          data[idx] = netParams.layer(id).blobs(0).data(idx);
        }

        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.ptr = data;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.bufSize = dataSize;

      }
    }
  }
  if(paramSet == 0)
  {
    dataSize =
    ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1])/
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups);
    printf("Setting RAND Kernel Params for Layer %s \n",
    (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);


    float  * data   = (float *)malloc(dataSize*sizeof(float));
    for (int idx = 0; idx < dataSize; idx++) {
      uint8_t val = (rand() & (0X7F));
      data[idx] = ((float)((rand()&1) ? val : -val))/64;
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.ptr = data;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.bufSize = dataSize;
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias =
  netStructure.layer(i).convolution_param().bias_term();
  if(pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias)
  {
    paramSet = 0;
    if(id != -1)
    {
      if(netParams.layer(id).blobs_size() > 1)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias = 1;
        dataSize = netParams.layer(id).blobs(1).data_size();
        if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]))
        {
          printf("Bias Size not matching!!");

        }
        else
        {
          paramSet = 1;
          float * data = (float *)malloc(dataSize*sizeof(float));
          for (int idx = 0; idx < dataSize; idx++)
          {
            data[idx] = netParams.layer(id).blobs(1).data(idx);
          }
          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.ptr = data;
          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.bufSize = dataSize;
        }
      }
    }
    if(paramSet == 0)
    {
      dataSize =
      (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]);
      printf("Setting RAND BIAS Params for Layer %s \n",
      (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);

      float  * data   = (float *)malloc(dataSize*sizeof(float));
      for (int idx = 0; idx < dataSize; idx++) {
        uint16_t val = (rand() & (0X7FFF));
        data[idx] = ((float)((rand()&1) ? val : -val))/256;
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.ptr = data;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.bufSize = dataSize;
    }
  }
  if(pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.inDataQ < 0)
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.inDataQ = 0;
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  (int64_t)(((int64_t)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]) / pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups);
}

void TIDL_importPoolingParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t              i,
int32_t              *pLayerIndex,
int32_t              *pDataIndex,
NetParameter         netStructure)
{
  int32_t     status;
  int32_t     layerIndex;
  int32_t     dataIndex;

  layerIndex = *pLayerIndex;
  dataIndex  = *pDataIndex;
  if(TIDL_isInputConv2D( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).bottom(0).c_str()) &&
      (netStructure.layer(i).pooling_param().kernel_size() == 2) &&
      (netStructure.layer(i).pooling_param().stride() == 2) &&
	  (TIDL_isBottomUsedLater(netStructure.layer(i).bottom(0).c_str(),i,netStructure) == false))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enablePooling = 1;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.poolingType = netStructure.layer(i).pooling_param().pool();
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.kernelW   = 2;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.kernelH   = 2;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.strideW   = 2;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.strideH   = 2;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outDataNames[0],netStructure.layer(i).top(0).c_str());

    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).pooling_param().pool() == PoolingParameter_PoolMethod_AVE)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.outDataQ = netStructure.layer(i).quantization_param().fl_layer_out();
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.poolParams.outDataQ = 1;
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[2] /= 2;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[3] /= 2;
  }
  else
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_PoolingLayer;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.poolingType    = netStructure.layer(i).pooling_param().pool();

    if(netStructure.layer(i).pooling_param().global_pooling() == true)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelW = 0;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelH = 0;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelW        = netStructure.layer(i).pooling_param().kernel_size();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelH        = netStructure.layer(i).pooling_param().kernel_size();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.strideW        = netStructure.layer(i).pooling_param().stride();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.strideH        = netStructure.layer(i).pooling_param().stride();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padW           = netStructure.layer(i).pooling_param().pad();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padH           = netStructure.layer(i).pooling_param().pad();
    }
#if 0
    if(netStructure.layer(i).pooling_param().kernel_size() > netStructure.layer(i).pooling_param().stride())
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padW  = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelW -1)/2;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padH  = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelH -1)/2;
    }
#endif

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
    if(netStructure.layer(i).pooling_param().global_pooling() == true)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]  = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2]  = 1;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]  = 1;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]  = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = ceil(((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] +
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padH*2.0) -
      (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelH))/
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.strideH) + 1;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = ceil(((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3] +
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.padW*2.0) -
      (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelW))/
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.strideW) + 1;
    }
     pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.numChannels =  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).pooling_param().pool() == PoolingParameter_PoolMethod_AVE)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.inDataQ  =  netStructure.layer(i).quantization_param().fl_layer_in(0);
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.inDataQ = 1;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.outDataQ = 1;
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = pOrgTIDLNetStructure->  TIDLPCLayers[layerIndex].inData[0].elementType;
    }


    if(netStructure.layer(i).pooling_param().global_pooling() == false)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]*
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelW *
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.poolParams.kernelH ;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]*
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];
   }

    layerIndex++;
  }
  *pLayerIndex = layerIndex;
  *pDataIndex = dataIndex ;
}

void TIDL_foldBNToConv2D(
float    * conv2weights,
float    * conv2dBias,
uint32_t  kernerlSize,
uint32_t  numCh,
float * mean,
float * var,
float * scale,
float * bias,
float eps
)
{
  kernerlSize /= numCh;
  uint32_t i, j;
  for(j = 0; j < numCh; j++)
  {
      float cb = conv2dBias[j];
      float m = mean[j];
      float v = var[j];
      float s = scale[j];
      float b = bias[j];
      double inv_var = pow((eps + v),-0.5);
      for(i = 0; i < kernerlSize; i++)
      {
        float w = conv2weights[j*kernerlSize + i];
        conv2weights[j*kernerlSize + i] = (w*s)*inv_var;
      }
      conv2dBias[j] = (((cb-m)*s)*inv_var) + b;
  }
}

void TIDL_importPRelUParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t              i,
int32_t              *pLayerIndex,
int32_t              *pDataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t     status;
  int32_t     layerIndex;
  int32_t     dataIndex;
  int32_t     channel_shared;
  int32_t     paramSet = 0;
  int32_t     dataSize;
   int32_t	  id;

  layerIndex = *pLayerIndex;
  dataIndex  = *pDataIndex;
  if(TIDL_isInputBatchNorm(pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).top(0).c_str()))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.enableRelU = 0;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.reluType = TIDL_PRelU;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outDataNames[0],netStructure.layer(i).top(0).c_str());
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
    }

    dataSize = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].inData[0].dimValues[1];
    if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[1]))
    {
      printf("PRelU Size not matching!!");
    }
    else
    {
      paramSet = 1;
      float * data = (float *)malloc(dataSize*sizeof(float));
      channel_shared = netStructure.layer(i).prelu_param().channel_shared();
	    id = TIDL_appCNNInteropCaffeFindLayerByName((char*)netStructure.layer(i).name().c_str(),netParams);
      for (int idx = 0; idx < dataSize; idx++)
      {
        if(channel_shared)
        {
          data[idx] = netParams.layer(id).blobs(0).data(0);
        }
        else
        {
          data[idx] = netParams.layer(id).blobs(0).data(idx);
        }
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.slope.ptr = data;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.slope.bufSize = dataSize;
    }
    if(paramSet == 0)
    {
      dataSize =
      (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[1]);
      printf("Setting RAND SLOPE Params for Layer %s \n",
      (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].name);
      float  * data   = (float *)malloc(dataSize*sizeof(float));
      for (int idx = 0; idx < dataSize; idx++) {
        uint16_t val = (rand() & (0X7FFF));
        data[idx] = ((float)((rand()&1) ? val : -val))/256;
      }
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.slope.ptr = data;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.slope.bufSize = dataSize;
    }
  }
  else
  {
    id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
    if(id == -1)
    {
      printf("Could not file BN Params for layer name %s\n",(char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);\
      exit(0);
    }
    uint32_t j, dataSize = netParams.layer(id).blobs(0).data_size();
    float       eps = 0;
    float * mean  = (float*)malloc(dataSize*sizeof(float));
    float * var   = (float*)malloc(dataSize*sizeof(float));
    float * scale = (float*)malloc(dataSize*sizeof(float));
    float * bias  = (float*)malloc(dataSize*sizeof(float));

     for(j = 0; j < dataSize; j++)
     {
       mean[j]  = 0;
       var[j]   = 1;
       scale[j] = 1;
       bias[j]  = 0;
     }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_BatchNormLayer;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.enableRelU = 0;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.reluType = TIDL_PRelU;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType  = TIDL_SignedChar;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];

    if(dataSize != pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1])
    {
      printf("Un-suported data size for BN\n");
    }
    else
    {
       float * dataBias    = (float*) malloc(dataSize*sizeof(float));
       float * dataWeigths = (float*) malloc(dataSize*sizeof(float));
       for(j = 0; j < dataSize; j++)
       {
         dataBias[j]  = 0;
         dataWeigths[j]  =  1;
       }
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr = dataBias;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize = dataSize;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr = dataWeigths;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize = dataSize;

       TIDL_foldBNToConv2D(
       (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr,
       (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr,
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize,
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize,
       mean,var,scale,bias,eps);
    }

    if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]))
    {
      printf("PRelU Size not matching!!");
    }
    else
    {
      paramSet = 1;
      float * data = (float *)malloc(dataSize*sizeof(float));
      channel_shared = netStructure.layer(i).prelu_param().channel_shared();
	    id = TIDL_appCNNInteropCaffeFindLayerByName((char*)netStructure.layer(i).name().c_str(),netParams);
      for (int idx = 0; idx < dataSize; idx++)
      {
        if(channel_shared)
        {
          data[idx] = netParams.layer(id).blobs(0).data(0);
        }
        else
        {
          data[idx] = netParams.layer(id).blobs(0).data(idx);
        }
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.slope.ptr = data;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.slope.bufSize = dataSize;
    }
    if(paramSet == 0)
    {
      dataSize =
      (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]);
      printf("Setting RAND SLOPE Params for Layer %s \n",
      (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);
      float  * data   = (float *)malloc(dataSize*sizeof(float));
      for (int idx = 0; idx < dataSize; idx++) {
        uint16_t val = (rand() & (0X7FFF));
        data[idx] = ((float)((rand()&1) ? val : -val))/256;
      }
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.slope.ptr = data;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.slope.bufSize = dataSize;
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];
    layerIndex++;

  }
  *pLayerIndex = layerIndex;
  *pDataIndex = dataIndex ;
}


void TIDL_importRelUParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t              i,
int32_t              *pLayerIndex,
int32_t              *pDataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t     status;
  int32_t     layerIndex;
  int32_t     dataIndex;
   int32_t	  id;

  layerIndex = *pLayerIndex;
  dataIndex  = *pDataIndex;
  if(TIDL_isInputConv2D( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).bottom(0).c_str()))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enableRelU = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outDataNames[0],netStructure.layer(i).top(0).c_str());
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).quantization_param().quantize_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
    }

  }
  else if(TIDL_isInputEltWise( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).top(0).c_str()))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.eltWiseParams.enableRelU = 1;
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).quantization_param().quantize_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.eltWiseParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
    }
  }
  else if(TIDL_isInputInnreProduct( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).top(0).c_str()))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.innerProductParams.enableRelU = 1;
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).quantization_param().quantize_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.innerProductParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
    }
  }
  else if(TIDL_isInputBatchNorm(pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).top(0).c_str()))
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.enableRelU = 1;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.reluParams.reluType = TIDL_RelU;
    if(quantizationStyle == TIDL_quantStyleFixed)
    {
#if ENABLE_FIXED_QUANT_STYLE
      if(netStructure.layer(i).quantization_param().quantize_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.batchNormParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
      }
      if(netStructure.layer(i).quantization_param().unsigned_layer_out())
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
      }
      else
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_SignedChar;
      }
#endif
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].elementType = TIDL_UnsignedChar;
    }
  }
  else
  {
    id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
    if(id == -1)
    {
      printf("Could not file BN Params\n");\
      exit(0);
    }
    uint32_t j, dataSize;
    float       eps = 0;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_BatchNormLayer;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.enableRelU = 1;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.reluType = TIDL_RelU;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType  = TIDL_UnsignedChar;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];

    dataSize = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    {
      float * dataBias    = (float*) malloc(dataSize*sizeof(float));
      float * dataWeigths = (float*) malloc(dataSize*sizeof(float));
      float * mean  = (float*)malloc(dataSize*sizeof(float));
      float * var   = (float*)malloc(dataSize*sizeof(float));
      float * scale = (float*)malloc(dataSize*sizeof(float));
      float * bias  = (float*)malloc(dataSize*sizeof(float));
      for(j = 0; j < dataSize; j++)
      {
        dataBias[j]  = 0;
        dataWeigths[j]  =  1;
        mean[j]  = 0;
        var[j]   = 1;
        scale[j] = 1;
        bias[j]  = 0;
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr = dataBias;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize = dataSize;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr = dataWeigths;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize = dataSize;

      TIDL_foldBNToConv2D(
      (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr,
      (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr,
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize,
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize,
      mean,var,scale,bias,eps);
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];
    layerIndex++;
  }
  *pLayerIndex = layerIndex;
  *pDataIndex = dataIndex ;
}

void TIDL_importDropoutParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t              i,
int32_t              layerIndex,
int32_t              dataIndex,
NetParameter         netStructure)
{
  int32_t     status;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_DropOutLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0];
}

void TIDL_importSoftmaxParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               layerIndex,
int32_t               dataIndex,
NetParameter          netStructure,
NetParameter          netParams)
{
  int32_t     id, status;
  int         prevLayerIdx = 0;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_SoftMaxLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);

  prevLayerIdx = TIDL_inputLayerIndex(pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).bottom(0).c_str());

  if(quantizationStyle == TIDL_quantStyleFixed)
  {
#if ENABLE_FIXED_QUANT_STYLE
    id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[prevLayerIdx].name,netParams);


    if(id != -1)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.softMaxParams.inDataQ = netParams.layer(id).quantization_param().fl_layer_out();
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.softMaxParams.outDataQ = netParams.layer(id).quantization_param().fl_layer_out();
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.softMaxParams.inDataQ  = 5;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.softMaxParams.outDataQ = 5;
    }
#endif
  }

  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];
}

void TIDL_importDeconvParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
FILE                 *fp1,
int32_t              i,
int32_t              layerIndex,
int32_t              dataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t             status, id;
  int32_t             pad, stride;
  int32_t             paramSet = 0;
  int32_t             dataSize;
  int32_t             conv2DRandParams = 0;
  float min, max;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_Deconv2DLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numInChannels  = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numOutChannels = netStructure.layer(i).convolution_param().num_output();
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups      = netStructure.layer(i).convolution_param().group();
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW        = netStructure.layer(i).convolution_param().kernel_size(0);
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH        = netStructure.layer(i).convolution_param().kernel_size(0);

  if (netStructure.layer(i).convolution_param().pad_size() == 0) {
    pad = 0;
  }
  else {
    pad = netStructure.layer(i).convolution_param().pad(0);
  }
  if (netStructure.layer(i).convolution_param().stride_size() == 0) {
    stride = 1;
  }
  else {
    stride = netStructure.layer(i).convolution_param().stride(0);
  }

  if (netStructure.layer(i).convolution_param().dilation_size() == 0) {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW = 1;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH = 1;
  }
  else {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW = netStructure.layer(i).convolution_param().dilation(0);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideW        = stride;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideH        = stride;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padW           = pad;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padH           = pad;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numOutChannels;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] =
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] - 1) *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideH +
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH-1)*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationH + 1) -
  (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padH*2));
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] =
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3] - 1) *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.strideW +
  ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW-1)*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.dilationW + 1) -
  (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.padW*2));

  if(quantizationStyle == TIDL_quantStyleFixed)
  {
#if ENABLE_FIXED_QUANT_STYLE
    if(netStructure.layer(i).quantization_param().unsigned_layer_out())
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_UnsignedChar;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_SignedChar;
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.inDataQ =  netStructure.layer(i).quantization_param().fl_layer_in(0);
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weightsQ =  netStructure.layer(i).quantization_param().fl_weights();
#endif
  }
  else
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_UnsignedChar;
  }

  id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  paramSet = 0;
  if(id != -1)
  {
    if(netParams.layer(id).blobs_size() > 0)
    {
      dataSize = netParams.layer(id).blobs(0).data_size();
      if(dataSize != ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
              pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1])/
            pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups))
      {
        printf("Kernel Size not matching!!");
      }
      else
      {
        paramSet = 1;
        float  * data   = (float *)malloc(dataSize*sizeof(float));
        for (int idx = 0; idx < dataSize; idx++)
        {
          data[idx] = netParams.layer(id).blobs(0).data(idx);
        }
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.ptr = data;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.bufSize = dataSize;
      }
    }
  }
  if(paramSet == 0)
  {
    printf("Setting RAND Kernel Params for Layer %s \n", (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);
    dataSize = ((pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1])/
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups);

    float  * data   = (float *)malloc(dataSize*sizeof(float));
    for (int idx = 0; idx < dataSize; idx++) {
      uint8_t val = (rand() & (0X7F));
      data[idx] = ((float)((rand()&1) ? val : -val))/64;
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.ptr = data;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.weights.bufSize = dataSize;
  }

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias = netStructure.layer(i).convolution_param().bias_term();

  if(pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias)
  {
    paramSet = 0;
    if(id != -1)
    {
      if(netParams.layer(id).blobs_size() > 1)
      {
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.enableBias = 1;
        dataSize = netParams.layer(id).blobs(1).data_size();
        if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]))
        {
          printf("Bias Size not matching!!");
        }
        else
        {
          paramSet = 1;
          float * data = (float *)malloc(dataSize*sizeof(float));
          for (int idx = 0; idx < dataSize; idx++)
          {
            data[idx] = netParams.layer(id).blobs(1).data(idx);
          }


          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.ptr = data;
          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.bufSize = dataSize;


        }
      }

    }
    if(paramSet == 0)
    {
      printf("Setting RAND BIAS Params for Layer %s \n", (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);

      dataSize = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]);
      float  * data   = (float *)malloc(dataSize*sizeof(float));
      for (int idx = 0; idx < dataSize; idx++) {
        uint8_t val = (rand() & (0X7F));
        data[idx] = ((float)((rand()&1) ? val : -val))/64;
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.ptr = data;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.bias.bufSize = dataSize;
    }
  }


  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  (int64_t)(((int64_t)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelW *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.kernelH *
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]) / pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.convParams.numGroups);
}


void TIDL_importArgmaxParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               layerIndex,
int32_t               dataIndex,
NetParameter          netStructure)
{
  int32_t     status;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_ArgMaxLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.argMaxParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_UnsignedChar;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = 1;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  (int64_t)((int64_t)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3])*2;

}


void TIDL_importBiasParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
FILE                 *fp1,
int32_t              i,
int32_t              layerIndex,
int32_t              dataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t             status, id;
  int32_t             paramSet = 0;
  int32_t             dataSize;
  int32_t             conv2DRandParams = 0;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_BiasLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.biasParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_SignedChar;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];


  id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  paramSet = 0;
  if(id != -1)
  {
    if(netParams.layer(id).blobs_size() > 0)
    {
      dataSize = netParams.layer(id).blobs(0).data_size();
      if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]))
      {
        printf("Bias Size not matching!!");
        paramSet = 0;
      }
      else
      {
        paramSet = 1;
        int16_t * params = (int16_t *)malloc(dataSize*2);
        for (int idx = 0; idx < dataSize; idx++) {
          float data = netParams.layer(id).blobs(0).data(idx);
          params[idx] = data * (1 << (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.biasParams.biasQ));
        }
        fwrite(params,2,dataSize,fp1);
        free(params);
      }
    }
  }
  if(paramSet == 0)
  {
    printf("Setting RAND BIAS Params for Layer %s \n", (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);
    dataSize = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]);
    int16_t * params = (int16_t *)malloc(dataSize*2);
    for (int idx = 0; idx < dataSize; idx++) {
      uint16_t val = (rand() & (0X7FFF));
      //params[idx] = (rand()&1) ? val : -val;
      params[idx] = -128;
    }
    fwrite(params,2,dataSize,fp1);
    free(params);
  }

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  (int64_t)((int64_t)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3]);

}


void TIDL_importEltwiseParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               layerIndex,
int32_t               dataIndex,
NetParameter          netStructure)
{
  int32_t     status;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_EltWiseLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = netStructure.layer(i).bottom_size();
  for (int32_t j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs; j++)
  {
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[j],netStructure.layer(i).bottom(j).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[j], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[j]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.eltWiseType        = netStructure.layer(i).eltwise_param().operation();
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.numInData          = netStructure.layer(i).bottom_size();
  if(quantizationStyle == TIDL_quantStyleFixed)
  {
#if ENABLE_FIXED_QUANT_STYLE
    if(netStructure.layer(i).quantization_param().unsigned_layer_out())
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_UnsignedChar;
    }
    else
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType = TIDL_SignedChar;
    }
    for(int32_t j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.numInData; j ++)
    {
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.inDataQ[j] =  netStructure.layer(i).quantization_param().fl_layer_in(j);
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.outDataQ   =  netStructure.layer(i).quantization_param().fl_layer_out();
#endif
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.eltWiseParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1];

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];

}


void TIDL_foldScaleToConv2D(
float    * conv2weights,
float    * conv2dBias,
uint32_t  kernerlSize,
uint32_t  numCh,
float * scale,
float * bias
)
{
  kernerlSize /= numCh;
  uint32_t i, j;
  for(j = 0; j < numCh; j++)
  {
      float cb = conv2dBias[j];
      float s = scale[j];
      float b = bias[j];
      for(i = 0; i < kernerlSize; i++)
      {
        float w = conv2weights[j*kernerlSize + i];
        conv2weights[j*kernerlSize + i] = (w*s);
      }
      conv2dBias[j] = (cb*s) + b;
  }
}
void TIDL_importScaleParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               *pLayerIndex,
int32_t               *pDataIndex,
NetParameter          netStructure,
NetParameter         netParams)
{
  int32_t     status;
  int32_t id;
  int32_t layerIndex = *pLayerIndex;
  int32_t dataIndex  = *pDataIndex;
  int32_t j;



  id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  if(id == -1)
  {
    printf("Could not file Scale Params\n");\
    exit(0);
  }
  uint32_t dataSize = netParams.layer(id).blobs(0).data_size();

  float * scale = (float*)malloc(dataSize*sizeof(float));
  float * bias  = (float*)malloc(dataSize*sizeof(float));
  if(netParams.layer(id).blobs_size() == 2)
  {
       for(j = 0; j < dataSize; j++)
       {
         scale[j]  = netParams.layer(id).blobs(0).data(j);
         bias[j]   = netParams.layer(id).blobs(1).data(j);
       }
  }
  else
  {
    printf("Un-suported number of blobs for Scale\n");
  }

  if(TIDL_isInputConv2D( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).bottom(0).c_str()))
  {
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outDataNames[0],netStructure.layer(i).top(0).c_str());
    if(dataSize != pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[1])
    {
      printf("Un-suported data size for Scale\n");
    }
    if(pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enableBias == 0)
    {
       float * data = (float*) malloc(dataSize*sizeof(float));
       for(j = 0; j < dataSize; j++)
       {
         data[j]  = 0;
       }
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enableBias = 1;
     }

     TIDL_foldScaleToConv2D(
     (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.weights.ptr,
     (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.bias.ptr,
     pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.weights.bufSize,
     dataSize,
     scale,bias);
  }
  else
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_ScaleLayer;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];
    layerIndex++;
  }
  *pLayerIndex = layerIndex;
  *pDataIndex = dataIndex ;


}



void TIDL_importBatchNormParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               *pLayerIndex,
int32_t               *pDataIndex,
NetParameter          netStructure,
NetParameter         netParams,
int32_t              layerType)
{
  int32_t     status;
  int32_t id;
  int32_t layerIndex = *pLayerIndex;
  int32_t dataIndex  = *pDataIndex;
  int32_t j;
  float eps = 0;
  uint32_t dataSize;


  id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  if(id == -1)
  {
    printf("Could not file BN Params\n");
    dataSize = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  }
  else
  {
    dataSize = netParams.layer(id).blobs(0).data_size();
    //
    // printf("layerIndex %d, dataSize for batchnorm layer %d: %d\n", layerIndex, id,dataSize);
  }

  float * mean  = (float*)malloc(dataSize*sizeof(float));
  float * var   = (float*)malloc(dataSize*sizeof(float));
  float * scale = (float*)malloc(dataSize*sizeof(float));
  float * bias  = (float*)malloc(dataSize*sizeof(float));

  if(id == -1)
  {
       for(j = 0; j < dataSize; j++)
       {
         mean[j]  = 0;
         var[j]   = 1;
         scale[j] = 1;
         bias[j]  = 0;
       }
  }
  else
  {

    if(layerType == 0)//BatchNorm
    {
      eps = netParams.layer(id).batch_norm_param().eps();
      if(netParams.layer(id).blobs_size() == 5)
      {

        // old format: 0 - scale , 1 - bias,  2 - mean , 3 - var, 4 - reserved
        // new format: 0 - mean  , 1 - var,  2 - reserved , 3- scale, 4 - bias
        if(netParams.layer(id).blobs(4).data_size() == 1)
        {
           for(j = 0; j < dataSize; j++)
           {
             mean[j]  = netParams.layer(id).blobs(2).data(j);
             var[j]   = netParams.layer(id).blobs(3).data(j);
             scale[j] = netParams.layer(id).blobs(0).data(j);
             bias[j]  = netParams.layer(id).blobs(1).data(j);
           }
        }
        else
        {
           for(j = 0; j < dataSize; j++)
           {
             mean[j]  = netParams.layer(id).blobs(0).data(j);
             var[j]   = netParams.layer(id).blobs(1).data(j);
             scale[j] = netParams.layer(id).blobs(3).data(j);
             bias[j]  = netParams.layer(id).blobs(4).data(j);
           }
        }
      }
      else if(netParams.layer(id).blobs_size() == 3)
      {
           for(j = 0; j < dataSize; j++)
           {
             mean[j]  = netParams.layer(id).blobs(0).data(j)/netParams.layer(id).blobs(2).data(0);
             var[j]   = netParams.layer(id).blobs(1).data(j)/netParams.layer(id).blobs(2).data(0);
             scale[j] = 1;
             bias[j]  = 0;
           }
      }
      else
      {
        printf("Un-suported number of blobs for BN, datasize = %d\n",dataSize);
      }
    }
    else if(layerType == 1)//bias
    {
         for(j = 0; j < dataSize; j++)
         {
           mean[j]  = 0;
           var[j]   = 1;
           scale[j] = 1;
           bias[j]  = netParams.layer(id).blobs(0).data(j);
         }
    }
    else if(layerType == 2)//scale
    {
      // printf("Layer num: %d, data size %d, checkpoint (5)\n",i,dataSize);
         for(j = 0; j < dataSize; j++)
         {
           mean[j]  = 0;
           var[j]   = 1;
           scale[j] = netParams.layer(id).blobs(0).data(j);
           // bias[j]  = netParams.layer(id).blobs(2).data(j);
           bias[j]  = netParams.layer(id).blobs(1).data(j);
         }
         // printf("Layer num: %d, data size %d, checkpoint (6)\n",i,dataSize);
    }
  }


  if((gParams.foldBnInConv2D == 1) && (TIDL_isInputConv2D( pOrgTIDLNetStructure, layerIndex, netStructure.layer(i).bottom(0).c_str())))
  {
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outDataNames[0],netStructure.layer(i).top(0).c_str());
    if(dataSize != pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].outData[0].dimValues[1])
    {
      printf("Un-suported data size for BN, dasize=%d\n",dataSize);
    }
    if(pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enableBias == 0)
    {
       float * data = (float*) malloc(dataSize*sizeof(float));
       for(j = 0; j < dataSize; j++)
       {
         data[j]  = 0;
       }
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.bias.ptr = data;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.bias.bufSize = dataSize;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.enableBias = 1;
     }
     pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.reluParams.reluType = TIDL_RelU;
     TIDL_foldBNToConv2D(
     (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.weights.ptr,
     (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.bias.ptr,
     pOrgTIDLNetStructure->TIDLPCLayers[layerIndex-1].layerParams.convParams.weights.bufSize,
     dataSize,
     mean,var,scale,bias,eps);
  }
  else
  {
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_BatchNormLayer;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
    status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    if(status == -1)
    {
      printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
      exit(-1);
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.numChannels = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.reluParams.reluType = TIDL_RelU;

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType  = TIDL_SignedChar;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2];
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];

    if(dataSize != pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1])
    {
      printf("layer %d, name %s, Un-suported data size for BN,ds=%d,dimvalue=%d\n",i,netStructure.layer(i).name().c_str(),
      dataSize,pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]);
    }
    else
    {
       float * dataBias    = (float*) malloc(dataSize*sizeof(float));
       float * dataWeigths = (float*) malloc(dataSize*sizeof(float));
       for(j = 0; j < dataSize; j++)
       {
         dataBias[j]  = 0;
         dataWeigths[j]  =  1;
       }
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr = dataBias;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize = dataSize;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr = dataWeigths;
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize = dataSize;

       TIDL_foldBNToConv2D(
       (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.ptr,
       (float *)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.ptr,
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.weights.bufSize,
       pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.batchNormParams.bias.bufSize,
       mean,var,scale,bias,eps);
    }

    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];
    layerIndex++;
  }
  *pLayerIndex = layerIndex;
  *pDataIndex = dataIndex ;
}


void TIDL_importInnerProductParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
FILE                 *fp1,
int32_t              i,
int32_t              layerIndex,
int32_t              dataIndex,
NetParameter         netStructure,
NetParameter         netParams)
{
  int32_t             status, id;
  int32_t             paramSet = 0;
  int32_t             dataSize;
  float min, max;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_InnerProductLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[0],netStructure.layer(i).top(0).c_str());
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dataId = dataIndex++;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].elementType  = TIDL_SignedChar;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].numDim       = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].numDim;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1] = 1;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] = 1;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3] = netStructure.layer(i).inner_product_param().num_output();

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.numInNodes =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1];
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.numOutNodes =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3];

  if(quantizationStyle == TIDL_quantStyleFixed)
  {
 #if ENABLE_FIXED_QUANT_STYLE
   pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.inDataQ  =  netStructure.layer(i).quantization_param().fl_layer_in(0);
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.outDataQ =  netStructure.layer(i).quantization_param().fl_layer_out();
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.weightsQ =  netStructure.layer(i).quantization_param().fl_weights();
#endif
  }


  id = TIDL_appCNNInteropCaffeFindLayerByName((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name,netParams);
  paramSet = 0;
  if(id != -1)
  {
    if(netParams.layer(id).blobs_size() > 0)
    {
      dataSize = netParams.layer(id).blobs(0).data_size();
      {
        paramSet = 1;
        float  * data   = (float *)malloc(dataSize*sizeof(float));
        for (int idx = 0; idx < dataSize; idx++)
        {
          data[idx] = netParams.layer(id).blobs(0).data(idx);
        }

        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.weights.ptr = data;
        pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.weights.bufSize = dataSize;

      }
    }
  }
  if(paramSet == 0)
  {
    printf("Setting RAND Kernel Params for Layer %s \n", (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);
    dataSize = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.numInNodes*
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.numOutNodes );

    float  * data   = (float *)malloc(dataSize*sizeof(float));
    for (int idx = 0; idx < dataSize; idx++) {
      uint8_t val = (rand() & (0X7F));
      data[idx] = ((float)((rand()&1) ? val : -val))/64;
    }
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.weights.ptr = data;
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.weights.bufSize = dataSize;
  }

  //pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.enableBias =
  //netStructure.layer(i).inner_product_param().bias_term();

  if(netStructure.layer(i).inner_product_param().bias_term())
  {
    paramSet = 0;
    if(id != -1)
    {
      if(netParams.layer(id).blobs_size() > 1)
      {
        dataSize = netParams.layer(id).blobs(1).data_size();
        if(dataSize != (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]))
        {
          printf("Bias Size not matching!!");
        }
        else
        {
          paramSet = 1;
          float * data = (float *)malloc(dataSize*sizeof(float));
          for (int idx = 0; idx < dataSize; idx++)
          {
            data[idx] = netParams.layer(id).blobs(1).data(idx);
          }
          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.bias.ptr = data;
          pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.bias.bufSize = dataSize;
        }
      }
    }

    if(paramSet == 0)
    {
      printf("Setting RAND BIAS Params for Layer %s \n", (char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].name);

      dataSize = (pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]);

      float  * data   = (float *)malloc(dataSize*sizeof(float));
      for (int idx = 0; idx < dataSize; idx++) {
        uint16_t val = (rand() & (0X7FFF));
        data[idx] = ((float)((rand()&1) ? val : -val))/256;
      }
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.bias.ptr = data;
      pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerParams.innerProductParams.bias.bufSize = dataSize;
    }
  }
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numMacs =
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[0].dimValues[3]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[0] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[1]*
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[2] * pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0].dimValues[3];
}


void TIDL_importSplitParams(sTIDL_OrgNetwork_t   *pOrgTIDLNetStructure,
int32_t               i,
int32_t               layerIndex,
NetParameter          netStructure)
{
  int32_t     status;

  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].layerType =  TIDL_SplitLayer;
  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs = netStructure.layer(i).top_size();


  pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numInBufs  = 1;
  strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0],netStructure.layer(i).bottom(0).c_str());
  status = TIDL_getDataID(&pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0], pOrgTIDLNetStructure, layerIndex, pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
  if(status == -1)
  {
    printf("Could not find the requested input Data : %s !!",pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inDataNames[0]);
    exit(-1);
  }


  for (int32_t j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].numOutBufs; j++)
  {
    strcpy((char*)pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outDataNames[j],netStructure.layer(i).top(j).c_str());
    pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].outData[j] = pOrgTIDLNetStructure->TIDLPCLayers[layerIndex].inData[0];
  }
}

void TIDL_UpdateOutDataBuff(sTIDL_OrgNetwork_t * pOrgTIDLNetStructure, uint32_t numLayers, sTIDL_DataParams_t dataBuf)
{
    uint32_t i,j;
    for (i = 0; i < numLayers; i++)
    {
        for(j = 0; (j < pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs) && (pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs > 0 ); j++)
        {
          if(pOrgTIDLNetStructure->TIDLPCLayers[i].outData[j].dataId == dataBuf.dataId)
          {
              pOrgTIDLNetStructure->TIDLPCLayers[i].outData[j] = dataBuf;
          }
        }

    }

    return;
}



extern sTIDL_OrgNetwork_t      orgTIDLNetStructure;
extern sTIDL_Network_t         tIDLNetStructure;
void caffe_import( tidl_import_config * params)
{
  int32_t         i, j, layerNum;
  int32_t         layerIndex, tiLayerIndex;
  int32_t         dataIndex;
  int64_t         totalMacs = 0;
  const uint8_t   *name;
  NetParameter    netStructure;
  NetParameter    netParams;
  FILE            *fp1;
  int             paramSet = 0;
  int             conv2DRandParams = 0;
  int32_t weightsElementSizeInBits;
  int overWritefirstNode = 1 ;

  if((params->inWidth == -1) || (params->inHeight == -1) || (params->inNumChannels == -1))
  {
    overWritefirstNode = 0;
  }


  printf("Caffe Network File : %s  \n",(const char *)params->inputNetFile);
  printf("Caffe Model File   : %s  \n",(const char *)params->inputParamsFile);
  printf("TIDL Network File  : %s  \n",(const char *)params->outputNetFile);
  printf("TIDL Model File    : %s  \n",(const char *)params->outputParamsFile);


  quantizationStyle = params->quantizationStyle;
#if (!ENABLE_FIXED_QUANT_STYLE)
  if(quantizationStyle == TIDL_quantStyleFixed)
  {
    printf("Un Supported quantizationStyle : TIDL_quantStyleFixed\n");
    return ;
  }

#endif

//debug
  // printf("Reading file %s checkpoint (1)\n",(const char *)params->inputNetFile);
//debug
  TIDL_readProtoFromTextFile((const char *)params->inputNetFile, &netStructure);
  //debug
  // printf("Reading file %s checkpoint (2)\n",(const char *)params->inputParamsFile);
  //debug
  TIDL_readProtoFromBinaryFile((const char *)params->inputParamsFile, &netParams);
  //
  // printf("Reading file %s checkpoint (3)\n",(const char *)params->inputParamsFile);
  //
  TIDL_appCNNConverRawDataToData(netParams);
  //
  if(giopmodeltxtEn)
    Tidl_ExportBinaryDataToTxtData(netParams);
  // printf("Reading file %s checkpoint (4)\n",(const char *)params->inputParamsFile);
  //
  fp1 = fopen((const char *)params->outputParamsFile, "wb+");
  if(fp1 == NULL)
  {
    printf("Could not open %s file for writing \n",(const char *)params->outputParamsFile);
    exit(-1);
  }

  layerNum = netStructure.layer_size();
  name     = (uint8_t*)netStructure.name().c_str();
  if(netStructure.has_name())
  {
    printf("Name of the Network : %15s \n", netStructure.name().c_str());
  }
  printf("Num Inputs : %15d \n", netStructure.input_size());
  //
  // printf("Layer num: %d, checkpoint (1)\n",layerNum);
  //

  layerIndex = 0;
  dataIndex  = 0;
  if(netStructure.input_size())
  {
    for (i = 0; i < netStructure.input_size(); i++)
    {
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerType  = TIDL_DataLayer;
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].numInBufs  = -1;
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].numOutBufs = 1;

      strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].name,
      netStructure.input(i).c_str());
      strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].outDataNames[0],
      netStructure.input(i).c_str());
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dataId =
      dataIndex++;
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].numDim =
      netStructure.input_shape(i).dim_size();
      for (j = 0; j < netStructure.input_shape(i).dim_size(); j++)
      {
        orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[j] =
        netStructure.input_shape(i).dim(j);
        //
        // printf("Inputlayer shape %d, dim %d,dimvalue %d\n", i,j,netStructure.input_shape(i).dim(j));
      }
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].elementType =
      gParams.inElementType;
      // Jacinto-net and Sqeeze net
      orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dataQ = gParams.inQuantFactor;
      // Mobile Net
      //orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dataQ = 30*255;

      layerIndex++;
    }

  }
  else
  {
    printf("Input layer(s) not Available.. Assuming below one Input Layer !!");
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerType  = TIDL_DataLayer;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].numInBufs  = -1;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].numOutBufs = 1;

    strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].name,"indata");
    strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].outDataNames[0],
    "indata");
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].numDim = 4;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dataId =
    dataIndex++;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[0] = 1;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[1] = 3;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[2] = 224;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[3] = 224;
    orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].elementType  =
    TIDL_UnsignedChar;
    layerIndex++;
  }

  if(overWritefirstNode)
  {
      for (i = layerIndex-1; i >= 0; i--)
      {
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[1] = params->inNumChannels;
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[2] = params->inHeight;
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[3] = params->inWidth;
      }
  }
  // printf("Layer num: %d, checkpoint (2)\n",layerNum);
  for (i = 0; i < layerNum; i++)
  {
    // if(i >= 0)

    strcpy((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].name,
    netStructure.layer(i).name().c_str());

	//Set the weights size in bits
	orgTIDLNetStructure.TIDLPCLayers[layerIndex].weightsElementSizeInBits = NUM_WHGT_BITS;
    // printf("Layer num: %d, checkpoint (3)\n",i);
    if (netStructure.layer(i).type() == "Concat")
    {

      TIDL_importConcatParams(&orgTIDLNetStructure, i, layerIndex, dataIndex,
      netStructure);
      layerIndex++;
      dataIndex++;
    }
    else if(netStructure.layer(i).type() == "Convolution")
    {
      // printf("Layer num: %d, checkpoint (3)\n",i);
      TIDL_importConvParams(&orgTIDLNetStructure, fp1, i, layerIndex, dataIndex,
      netStructure, netParams);
      // if (!strcmp((char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].name,"conv_1_conv2d")) {
      //   printf("\n---Current convolution dimensions:dimVAlues[2]:%d,padH*2:%d,kernelH:%d,dilationH:%d,strideH:%d\n",
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].inData[0].dimValues[2],
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerParams.convParams.padH*2,
      //   // netStructure.layer(i).convolution_param().kernel_size(0),
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerParams.convParams.kernelH,
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerParams.convParams.dilationH,
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].layerParams.convParams.strideH);
      //   printf("\nCurrent Layer %s output dimension: %d,%d,%d,%d---\n",
      //   (char*)orgTIDLNetStructure.TIDLPCLayers[layerIndex].name,
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[0],
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[1],
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[2],
      //   orgTIDLNetStructure.TIDLPCLayers[layerIndex].outData[0].dimValues[3]);
      // }
      layerIndex++;
      dataIndex++;
      // printf("Layer num: %d, checkpoint (4)\n",i);
    }
    else if(netStructure.layer(i).type() == "Pooling")
    {
      TIDL_importPoolingParams(&orgTIDLNetStructure, i, &layerIndex,
      &dataIndex, netStructure);
    }
    else if ((netStructure.layer(i).type() == "ReLU") ||
        (netStructure.layer(i).type() == "LRN"))
    {
      TIDL_importRelUParams(&orgTIDLNetStructure, i, &layerIndex,
      &dataIndex, netStructure, netParams);
    }
    else if (netStructure.layer(i).type() == "PReLU")
    {
      TIDL_importPRelUParams(&orgTIDLNetStructure, i, &layerIndex,
      &dataIndex, netStructure, netParams);
    }
    else if (netStructure.layer(i).type() == "Dropout")
    {
      TIDL_importDropoutParams(&orgTIDLNetStructure, i, layerIndex, dataIndex,
      netStructure);
      layerIndex++;
      dataIndex++;
    }
    else if ((netStructure.layer(i).type() == "Softmax") ||
        (netStructure.layer(i).type() == "softmax"))
    {
      TIDL_importSoftmaxParams(&orgTIDLNetStructure, i, layerIndex, dataIndex,
      netStructure, netParams);
      layerIndex++;
      dataIndex++;
    }
    else if (netStructure.layer(i).type() == "Deconvolution")
    {
      TIDL_importDeconvParams(&orgTIDLNetStructure, fp1, i, layerIndex,
      dataIndex, netStructure, netParams);
      layerIndex++;
      dataIndex++;
    }
    else if ((netStructure.layer(i).type() == "Argmax") ||
        (netStructure.layer(i).type() == "ArgMax"))
    {
      TIDL_importArgmaxParams(&orgTIDLNetStructure, i, layerIndex, dataIndex,
      netStructure);
      layerIndex++;
      dataIndex++;
    }
    else if (netStructure.layer(i).type() == "Bias")
    {
#if 0
      TIDL_importBiasParams(&orgTIDLNetStructure, fp1, i, layerIndex, dataIndex,
      netStructure, netParams);
      layerIndex++;
      dataIndex++;
#else
      TIDL_importBatchNormParams(&orgTIDLNetStructure, i, &layerIndex, &dataIndex,
      netStructure,netParams,1);
#endif
    }
    else if(netStructure.layer(i).type() == "Eltwise")
    {
      TIDL_importEltwiseParams(&orgTIDLNetStructure, i, layerIndex, dataIndex,
      netStructure);
      layerIndex++;
      dataIndex++;
    }
    else if (netStructure.layer(i).type() == "BatchNorm")
    {

      TIDL_importBatchNormParams(&orgTIDLNetStructure, i, &layerIndex, &dataIndex,
      netStructure,netParams,0);

    }
    else if (netStructure.layer(i).type() == "Scale")
    {
#if 0
      TIDL_importScaleParams(&orgTIDLNetStructure, i, &layerIndex, &dataIndex,
      netStructure,netParams);
#else
    // printf("Layer num: %d, checkpoint (4)\n",i);
      TIDL_importBatchNormParams(&orgTIDLNetStructure, i, &layerIndex, &dataIndex,
      netStructure,netParams,2);
      // netStructure,netParams,1);
#endif
      // printf("Layer num: %d, checkpoint (5)\n",i);
    }
    else if (netStructure.layer(i).type() == "InnerProduct")
    {
      TIDL_importInnerProductParams(&orgTIDLNetStructure, fp1, i, layerIndex,
      dataIndex, netStructure, netParams);
      layerIndex++;
      dataIndex++;
    }
    else if(netStructure.layer(i).type() == "Split")
    {
      TIDL_importSplitParams(&orgTIDLNetStructure, i, layerIndex,
      netStructure);
      layerIndex++;
    }
    else
    {
      printf("Unsuported Layer Type : %s !!!!\n",
      netStructure.layer(i).type().c_str());
    }
  }
  // printf("Layer num: %d, checkpoint (5)\n",layerNum);
    for (i = 0; i < layerIndex; i++)
    {
      /* Find Inner Product layers */
      if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer)
      {
        if((orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[2] == 1) &&
          (orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[3] == 1) &&
          (orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[2] == 1) &&
          (orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[3] == 1) &&
          (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.kernelW == 1) &&
          (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.kernelH == 1))
        {
          orgTIDLNetStructure.TIDLPCLayers[i].layerType = TIDL_InnerProductLayer;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.enableRelU = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.enableRelU;

          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.ptr = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.bufSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize;

          if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.enableBias)
          {
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr;
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize;
          }
          else
          {
            printf("TIDL_InnerProductLayer without Bias is not supported \n");
          }

          //orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr = NULL;
          //orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize = 0;
          //orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr = NULL;
          //orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize = 0;
        }
      }
    }

    for (i = 0; i < layerIndex; i++)
    {
      /* Find Convolution Layer iwht Just one input channel
      The minimum number of input channel required for TIDL convolution layer is 2
      So making it to 2 input channel and ans setting all the kernel co-efficents for
      second input channel as zero*/
      if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer)
      {
        if(orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[1] == 1)
        {
          orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[1] = 2;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.numInChannels = 2;
          {
          uint32_t k = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.kernelW *
                       orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.kernelH;
          uint32_t numOutCh = orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[1];

          float *data       = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize;

          float *outData       = (float *)malloc(dataSize*2*sizeof(float));

            for (int idx = 0; idx < numOutCh; idx++)
            {
              for (int idx2 = 0; idx2 < k; idx2++)
              {
                outData[2*idx*k + idx2] = data[idx*k + idx2];
                outData[2*idx*k + k + idx2] = 0;
              }
            }
            free(data);
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr = outData;
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize = dataSize*2;
        }
        TIDL_UpdateOutDataBuff(&orgTIDLNetStructure,i,orgTIDLNetStructure.TIDLPCLayers[i].inData[0]);


       }
     }
    }


  if(quantizationStyle == TIDL_quantStyleDynamic)
  {
    /* Dynamically Quantize Layer Params */
     uint32_t totalParamSize = 0;
    for (i = 0; i < layerIndex; i++)
    {
      if((orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_Deconv2DLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_BatchNormLayer)
          )
      {
        float min = FLT_MAX;
        float max = FLT_MIN;

        if((orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_Deconv2DLayer))
        {
          weightsElementSizeInBits = orgTIDLNetStructure.TIDLPCLayers[i].weightsElementSizeInBits;
          float *  data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize;
          uint8_t * params = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits-1)/8 + 1));
          TIDL_findRange(data, dataSize, &min , &max, 1.0);
          if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.enableBias)
          {
            float * biasData      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr;
            uint32_t biasDataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize;
            TIDL_findRange(biasData, biasDataSize, &min , &max, (1.0/(1 << (16-NUM_WHGT_BITS))));
          }

          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weightsQ =
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data,dataSize, min , max, weightsElementSizeInBits);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.zeroWeightValue =
          TIDL_normalize(-min, min , max);

          if(weightsElementSizeInBits > 8)
          {
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
          }
          else
          {
            fwrite(params,1,dataSize,fp1);
            totalParamSize += dataSize;
          }
          free(params);
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize = 0;
          if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.enableBias)
          {
            data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr;
            dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize;
            int16_t * params = (int16_t *)malloc(dataSize*2);

            for (int idx = 0; idx < dataSize; idx++)
            {
              int32_t biasParam = TIDL_normalize(data[idx], min , max);
              params[idx] = (int16_t)TIDL_roundSat(biasParam,0,SHRT_MIN,SHRT_MAX);
            }
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
            free(params);
            free(data);
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr = NULL;
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize = 0;
          }

        }
        else if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer)
        {
          weightsElementSizeInBits = orgTIDLNetStructure.TIDLPCLayers[i].weightsElementSizeInBits;
          float *  data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.bufSize;
          uint8_t * params = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits-1)/8 + 1));
          TIDL_findRange(data, dataSize, &min , &max, 1.0);
          {
            float * biasData      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr;
            uint32_t biasDataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize;
            TIDL_findRange(biasData, biasDataSize, &min , &max, (1.0/(1 << (16-NUM_WHGT_BITS))));
          }

          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weightsQ =
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data,dataSize, min , max, weightsElementSizeInBits);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.zeroWeightValue =
          TIDL_normalize(-min, min , max);
          if(weightsElementSizeInBits > 8)
          {
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
          }
          else
          {
            fwrite(params,1,dataSize,fp1);
            totalParamSize += dataSize;
          }
          free(params);
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.bufSize = 0;
          data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr;
          dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize;
          //if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.enableBias)
          {
            int16_t *params = (int16_t *)malloc(dataSize*2);
            for (int idx = 0; idx < dataSize; idx++)
            {
              int32_t biasParam = TIDL_normalize(data[idx], min , max);
              params[idx] = (int16_t)TIDL_roundSat(biasParam,0,SHRT_MIN,SHRT_MAX);
            }
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
            free(params);
          }
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize = 0;
        }
        else if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_BatchNormLayer)
        {
          weightsElementSizeInBits = orgTIDLNetStructure.TIDLPCLayers[i].weightsElementSizeInBits;
          float *  data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.weights.bufSize;
          uint8_t * params = (uint8_t *)malloc(dataSize * ((weightsElementSizeInBits-1)/8 + 1));
          TIDL_findRange(data, dataSize, &min , &max, 1.0);
          {
            float * biasData      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.ptr;
            uint32_t biasDataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.bufSize;
            TIDL_findRange(biasData, biasDataSize, &min , &max, (1.0/(1 << (16-NUM_WHGT_BITS))));
          }
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.weightsQ =
          TIDL_QuantizeUnsignedMax((uint8_t *)params, data,dataSize, min , max,weightsElementSizeInBits);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.zeroWeightValue =
          TIDL_normalize(-min, min , max);

          if(weightsElementSizeInBits > 8)
          {
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
          }
          else
          {
            fwrite(params,1,dataSize,fp1);
            totalParamSize += dataSize;
          }
          free(params);
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.weights.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.weights.bufSize = 0;
          data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.ptr;
          dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.bufSize;
          {
            int16_t *params = (int16_t *)malloc(dataSize*2);
            for (int idx = 0; idx < dataSize; idx++)
            {
              int32_t biasParam = TIDL_normalize(data[idx], min , max);
              params[idx] = (int16_t)TIDL_roundSat(biasParam,0,SHRT_MIN,SHRT_MAX);
            }
            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
            free(params);
          }
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.bias.bufSize = 0;
          if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.reluType == TIDL_PRelU)
          {
            float * slopeData      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slope.ptr;
            uint32_t slopeDataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slope.bufSize;
            uint8_t * params = (uint8_t *)malloc(slopeDataSize * ((weightsElementSizeInBits-1)/8 + 1));
            float min = FLT_MAX;
            float max = FLT_MIN;
            TIDL_findRange(slopeData, slopeDataSize, &min , &max, (1.0));
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slopeQ =
            TIDL_QuantizeUnsignedMax((uint8_t *)params, slopeData,slopeDataSize, min , max,weightsElementSizeInBits);
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slopeQ /= 256;

            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.zeroSlopeValue =
            TIDL_normalize(-min, min , max);

            if(weightsElementSizeInBits > 8)
            {
              fwrite(params,2,slopeDataSize,fp1);
              totalParamSize += 2*slopeDataSize;
            }
            else
            {
              fwrite(params,1,slopeDataSize,fp1);
              totalParamSize += slopeDataSize;
            }
            free(params);
            free(slopeData);
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slope.ptr = NULL;
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.batchNormParams.reluParams.slope.bufSize = 0;
          }
        }
      }
    }
  }
  else if(quantizationStyle == TIDL_quantStyleFixed)
  {
    /* Fixed Quantization of Layer Params */
     uint32_t totalParamSize = 0;
    for (i = 0; i < layerIndex; i++)
    {
      if((orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_Deconv2DLayer)
          )
      {
        if((orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_ConvolutionLayer) ||
          (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_Deconv2DLayer))
        {
          float *  data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize;
          uint8_t * params = (uint8_t *)malloc(dataSize);
          for (int idx = 0; idx < dataSize; idx++)
          {
            params[idx] = TIDL_flApply(data[idx],
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weightsQ);
          }

          fwrite(params,1,dataSize,fp1);
          totalParamSize += dataSize;
          free(params);
          free(data);

          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weights.bufSize = 0;
          if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.enableBias)
          {
            data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr;
            dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize;
            int16_t * params = (int16_t *)malloc(dataSize*2);
            for (int idx = 0; idx < dataSize; idx++)
            {
              params[idx] = TIDL_flApply(data[idx],
              (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.weightsQ +
              orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.inDataQ));
            }

            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
            free(params);
            free(data);
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.ptr = NULL;
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.convParams.bias.bufSize = 0;
          }

        }
        else if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer)
        {
          float *  data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.ptr;
          uint32_t dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.bufSize;
          uint8_t * params = (uint8_t *)malloc(dataSize);

          for (int idx = 0; idx < dataSize; idx++)
          {
            params[idx] = TIDL_flApply(data[idx],
            orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weightsQ);
          }

          fwrite(params,1,dataSize,fp1);
          totalParamSize += dataSize;
          free(params);
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weights.bufSize = 0;
          data      = (float *)orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr;
          dataSize = orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize;
          //if(orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.enableBias)
          {
            int16_t *params = (int16_t *)malloc(dataSize*2);

            for (int idx = 0; idx < dataSize; idx++)
            {
              params[idx] = TIDL_flApply(data[idx],
              (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.weightsQ +
              orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.inDataQ));
            }

            fwrite(params,2,dataSize,fp1);
            totalParamSize += 2*dataSize;
            free(params);
          }
          free(data);
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.ptr = NULL;
          orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.bias.bufSize = 0;
        }
      }
    }
  }
  else
  {
      printf("Unsuported quantizationStyle \n");
  }

    /* Re-shape layers */
  for (i = 0; i < layerIndex; i++)
  {
    if(((orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_PoolingLayer) &&
      (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.poolParams.kernelW == 0 ) &&
      (orgTIDLNetStructure.TIDLPCLayers[i].layerParams.poolParams.kernelH == 0))
      ||
      (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer)
      ||
      (orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_SoftMaxLayer))
    {
      orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[3] =
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[1]*
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[2]*
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[3];
      orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[1] = 1;
      orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[2] = 1;
      TIDL_UpdateInDataBuff(&orgTIDLNetStructure,layerIndex,orgTIDLNetStructure.TIDLPCLayers[i].outData[0]);
      if(orgTIDLNetStructure.TIDLPCLayers[i].layerType == TIDL_InnerProductLayer)
      {
        orgTIDLNetStructure.TIDLPCLayers[i].layerParams.innerProductParams.numInNodes =
        orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[3];
      }
    }
  }



  printf(" Num of Layer Detected : %3d \n",layerIndex);

  tIDLNetStructure.dataElementSize    = 1;   //Set to 1 for 8-bit data and to 2 for 16-bit data
  tIDLNetStructure.biasElementSize    = 2;
  tIDLNetStructure.weightsElementSize = ((weightsElementSizeInBits-1)/8 + 1); //1;
  tIDLNetStructure.slopeElementSize   = tIDLNetStructure.weightsElementSize;
  tIDLNetStructure.interElementSize   = 4;
  tIDLNetStructure.quantizationStyle  = quantizationStyle;
  tIDLNetStructure.strideOffsetMethod = TIDL_strideOffsetTopLeft;

  tiLayerIndex = 0;
  for (i = 0; i < layerIndex; i++)
  {
    if((orgTIDLNetStructure.TIDLPCLayers[i].layerType != TIDL_SplitLayer) &&
      (orgTIDLNetStructure.TIDLPCLayers[i].layerType !=TIDL_DropOutLayer))
    {
      tIDLNetStructure.TIDLLayers[tiLayerIndex].layerType   =
      orgTIDLNetStructure.TIDLPCLayers[i].layerType;
      tIDLNetStructure.TIDLLayers[tiLayerIndex].layerParams =
      orgTIDLNetStructure.TIDLPCLayers[i].layerParams;
      tIDLNetStructure.TIDLLayers[tiLayerIndex].numInBufs   =
      orgTIDLNetStructure.TIDLPCLayers[i].numInBufs;
      tIDLNetStructure.TIDLLayers[tiLayerIndex].numOutBufs  =
      orgTIDLNetStructure.TIDLPCLayers[i].numOutBufs;
      if(tIDLNetStructure.TIDLLayers[tiLayerIndex].layerType == TIDL_DataLayer)
      {
        tIDLNetStructure.TIDLLayers[tiLayerIndex].layersGroupId      = 0;
      }
      else
      {
        if(tiLayerIndex < 0)
        {
          tIDLNetStructure.TIDLLayers[tiLayerIndex].coreID             = 2;
          tIDLNetStructure.TIDLLayers[tiLayerIndex].layersGroupId      = 2;
        }
        else
        {
          tIDLNetStructure.TIDLLayers[tiLayerIndex].coreID             = 1;
          tIDLNetStructure.TIDLLayers[tiLayerIndex].layersGroupId      = 1;
        }
      }
      printf("%3d, %-30s, %-30s", i,
      TIDL_LayerString[orgTIDLNetStructure.TIDLPCLayers[i].layerType],
      orgTIDLNetStructure.TIDLPCLayers[i].name);
      printf("%3d, %3d ,%3d , ",
      tIDLNetStructure.TIDLLayers[tiLayerIndex].layersGroupId,
      orgTIDLNetStructure.TIDLPCLayers[i].numInBufs,
      orgTIDLNetStructure.TIDLPCLayers[i].numOutBufs);

      for (j = 0; j < orgTIDLNetStructure.TIDLPCLayers[i].numInBufs; j++)
      {
        printf("%3d ,",orgTIDLNetStructure.TIDLPCLayers[i].inData[j]);
        tIDLNetStructure.TIDLLayers[tiLayerIndex].inData[j]   =
        orgTIDLNetStructure.TIDLPCLayers[i].inData[j];
      }
      j = 0;
      if(orgTIDLNetStructure.TIDLPCLayers[i].numInBufs > 0)
      j = orgTIDLNetStructure.TIDLPCLayers[i].numInBufs;
      for (; j < 8; j++)
      {
        printf("  x ,");
      }
      printf("%3d ,",orgTIDLNetStructure.TIDLPCLayers[i].outData[0]);
      tIDLNetStructure.TIDLLayers[tiLayerIndex].outData[0]   =
      orgTIDLNetStructure.TIDLPCLayers[i].outData[0];

      tIDLNetStructure.TIDLLayers[tiLayerIndex].outData[0].minValue = (int)(0x7FFFFFFF);
      tIDLNetStructure.TIDLLayers[tiLayerIndex].outData[0].maxValue = (int)(0x80000000);

      for (j = 0; j < TIDL_DIM_MAX; j++)
      {
        printf("%5d ,",
        orgTIDLNetStructure.TIDLPCLayers[i].inData[0].dimValues[j]);
      }

      for (j = 0; j < TIDL_DIM_MAX; j++)
      {
        printf("%5d ,",
        orgTIDLNetStructure.TIDLPCLayers[i].outData[0].dimValues[j]);
      }
      printf("%10lld ,",orgTIDLNetStructure.TIDLPCLayers[i].numMacs);
      totalMacs += orgTIDLNetStructure.TIDLPCLayers[i].numMacs;
      printf("\n");
      tiLayerIndex++;
    }
  }

  for (i = 0; i < tiLayerIndex; i++)
  {
    sTIDL_Layer_t *tidlLayer;
    tidlLayer = &tIDLNetStructure.TIDLLayers[tiLayerIndex];
    tidlLayer->layerType   = TIDL_DataLayer;
    tidlLayer->numInBufs   = 0;
    tidlLayer->numOutBufs  = -1;
    tidlLayer->coreID      = 255;

    if(tIDLNetStructure.TIDLLayers[i].layerType != TIDL_DataLayer)
    {
      for (j = 0 ; j < tIDLNetStructure.TIDLLayers[i].numOutBufs; j++)
      {
        if(!TIDL_isDataBufUsed(tIDLNetStructure.TIDLLayers[i].outData[j].dataId,
              &tIDLNetStructure, tiLayerIndex))
        {
          tidlLayer->inData[tidlLayer->numInBufs] =
          tIDLNetStructure.TIDLLayers[i].outData[j];
          tidlLayer->numInBufs++;
        }
      }
    }
  }
  tIDLNetStructure.numLayers = tiLayerIndex + 1;
  printf("Total layers processed %d\n", tIDLNetStructure.numLayers);

  printf("Total Giga Macs : %4.4f\n", ((float)totalMacs/1000000000));
  printf("Total Giga Macs : %4.4f @15 fps\n", 15*((float)totalMacs/1000000000));
  printf("Total Giga Macs : %4.4f @30 fps\n", 30*((float)totalMacs/1000000000));
  if(fp1 != NULL)
  {
    fclose(fp1);
  }
  fp1 = fopen((const char *)params->outputNetFile, "wb+");
  if(fp1 == NULL)
  {
    printf("Could not open %s file for writing \n",(const char *)params->outputNetFile);
  }
  fwrite(&tIDLNetStructure,1,sizeof(tIDLNetStructure),fp1);
  if(fp1 != NULL)
  {
    fclose(fp1);
  }
}
