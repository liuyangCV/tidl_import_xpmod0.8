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
#include <cfloat>

using namespace std;
using ::google::protobuf::Message;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::FileOutputStream;
using ::google::protobuf::io::ZeroCopyInputStream;
using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::ZeroCopyOutputStream;
using ::google::protobuf::io::CodedOutputStream;

#include "ti_dl.h"
#include "tidl_import_config.h"
#include "tidl_import_common.h" 

#define QUAN_STYLE2_ROUND ((gParams.quantRoundAdd*1.0 / 100))

int32_t TIDL_QuantizeUnsignedMax(uint8_t * params, float * data, int32_t dataSize, float min, float max, int32_t weightsElementSizeInBits)

{
  int32_t i;
  float absRange = abs(max - min);

  float quantPrec = ((1.0*(1 << NUM_WHGT_BITS)) / absRange);
  float pData;
  int32_t param;

  int32_t offset;
  if(min  > 0)
  {
    offset = (min *  quantPrec + QUAN_STYLE2_ROUND);
  }
  else
  {
    offset = (min *  quantPrec - QUAN_STYLE2_ROUND);
  }

  //Convert float params to 8-bit or 16-bit
  if(weightsElementSizeInBits <= 8)
  {
    for(i = 0; i < dataSize; i++)
    {
      pData = data[i]; 
      if(pData  > 0)
      {
        param = (pData *  quantPrec + QUAN_STYLE2_ROUND);
      }
      else
      {
        param = (pData *  quantPrec - QUAN_STYLE2_ROUND);
      }
      param = param - offset;

      params[i] = param > ((1 << weightsElementSizeInBits) - 1) ? ((1 << weightsElementSizeInBits) - 1) : param;
    }
  }
  else
  {
	uint16_t *params16 = (uint16_t *)params;

  for(i = 0; i < dataSize; i++)
  {
    pData = data[i]; 
    if(pData  > 0)
    {
      param = (pData *  quantPrec + QUAN_STYLE2_ROUND);
    }
    else
    {
      param = (pData *  quantPrec - QUAN_STYLE2_ROUND);
    }
    param = param - offset;

      params16[i] = param > ((1 << weightsElementSizeInBits) - 1) ? ((1 << weightsElementSizeInBits) - 1) : param;
    }
  }
  return ((int32_t)(quantPrec*256));
}

int32_t TIDL_normalize(float data, float min, float max)
{
  int32_t param;
  float absRange = abs(max - min);
  float quantPrec = ((1.0*(1 << NUM_WHGT_BITS)) / absRange);
  if(data  > 0)
  {
    param = (data *  quantPrec + QUAN_STYLE2_ROUND);
  }
  else
  {
    param = (data *  quantPrec - QUAN_STYLE2_ROUND);
  }


  return param;
}

bool TIDL_readProtoFromTextFile(const char* fileName, Message* proto)
{
  int32_t           fd;
  bool              success;
  FileInputStream   *input;

  fd      = open(fileName, O_RDONLY);
  if(fd == NULL)
  {
    printf("Could not open prototext file for reading \n");
    exit(-1);
  }
  input   = new FileInputStream(fd);
  success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  if (!success) 
  {
    printf("ERROR: Reading text proto file\n");
  }
  return success;
}

#define APP_CNN_INTEROP_CAFFE_READ_BINARY_TOTAL_BYTE_LIMIT  2147483647
#define APP_CNN_INTEROP_CAFFE_READ_BINARY_WARNING_THRESHOLD 1073741824 
bool TIDL_readProtoFromBinaryFile(const char* fileName, Message* proto)
{
  int                   fd;
  ZeroCopyInputStream   *rawInput;
  CodedInputStream      *codedInput;
  bool                  success;

  fd         = open(fileName, O_BINARY);
  if(fd == NULL)
  {
    printf("Could not open caffe model for reading \n");
    exit(-1);
  }

  rawInput   = new FileInputStream(fd);
  codedInput = new CodedInputStream(rawInput);

  codedInput->SetTotalBytesLimit(
  APP_CNN_INTEROP_CAFFE_READ_BINARY_TOTAL_BYTE_LIMIT, 
  APP_CNN_INTEROP_CAFFE_READ_BINARY_WARNING_THRESHOLD
  );
  
  success = proto->ParseFromCodedStream(codedInput);
  delete codedInput;
  delete rawInput;
  close(fd);
  if (!success) 
  {
    printf("ERROR: Reading binary proto file\n");
  }
  return success;
}


int32_t TIDL_getDataID(sTIDL_DataParams_t *data, 
sTIDL_OrgNetwork_t * pOrgTIDLNetStructure,
int32_t            numLayer, 
int8_t             *bufName)
{
  int32_t i,j;
  for (i = (numLayer-1); i >= 0; i--)
  {
    for (j = 0; j < pOrgTIDLNetStructure->TIDLPCLayers[i].numOutBufs; j++)
    {       
      if(strcmp((const char*)bufName,
            (const char*)pOrgTIDLNetStructure->TIDLPCLayers[i].outDataNames[j])==0)
      {
        *data = pOrgTIDLNetStructure->TIDLPCLayers[i].outData[j];
        return 0 ;
      }
    }
  }
  return -1;
}

int32_t TIDL_isDataBufUsed(int32_t           dataId, 
sTIDL_Network_t   *pTIDLNetStructure,
int32_t           numLayer)
{
  int32_t i,j;
  for (i = 0 ; i < numLayer; i++)
  {
    for (j = 0; j < pTIDLNetStructure->TIDLLayers[i].numInBufs; j++)
    {       
      if(pTIDLNetStructure->TIDLLayers[i].inData[j].dataId == dataId)
      {
        return 1 ;
      }
    }
  }
  return 0;
}

int32_t TIDL_isInputConv2D(sTIDL_OrgNetwork_t   *pOrgTIDLNetStruct,
int32_t              numLayer, 
const char           *bufName)
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
            (pOrgTIDLNetStruct->TIDLPCLayers[i].layerType==TIDL_ConvolutionLayer))
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

void TIDL_UpdateInDataBuff(sTIDL_OrgNetwork_t * pOrgTIDLNetStructure, 
uint32_t numLayers, sTIDL_DataParams_t dataBuf)
{
    uint32_t i,j;
    for (i = 0; i < numLayers; i++) 
    {
        for(j = 0; (j < pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs) && 
        (pOrgTIDLNetStructure->TIDLPCLayers[i].numInBufs > 0 ); j++)
        {
          if(pOrgTIDLNetStructure->TIDLPCLayers[i].inData[j].dataId == 
              dataBuf.dataId)
          {
              pOrgTIDLNetStructure->TIDLPCLayers[i].inData[j] = dataBuf;
          }
        }

    }

    return;
}

void TIDL_findRange(float * data, int32_t dataSize, float * minOut, float * maxOut, float scale)
{
  float min = FLT_MAX;
  float max = FLT_MIN;
  int32_t i;
  for(i = 0; i < dataSize; i++)
  {
    min = ((data[i]*scale) < min) ? (data[i]*scale) : min;
    max = ((data[i]*scale) > max) ? (data[i]*scale) : max;
  }
  *minOut = (min < *minOut) ? min : *minOut;
  *maxOut = (max > *maxOut) ? max : *maxOut;
}

int64_t TIDL_roundSat(int64_t val, uint8_t bits, int64_t min, int64_t max)
{
  if(bits > 0)
  {
    val += (1U << (bits - 1U));
    val >>= bits;
  }
  val =  val < min ? min :  val;
  val =  val > max  ? max  :  val;
  return val;

}