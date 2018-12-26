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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



#include "tidl_import_config.h"
#include "ti_dl.h"

const char * TIDL_LayerString[] =
{
"TIDL_DataLayer           ",
"TIDL_ConvolutionLayer    ",
"TIDL_PoolingLayer        ",
"TIDL_ReLULayer           ",
"TIDL_PReLULayer          ",
"TIDL_EltWiseLayer        ",
"TIDL_InnerProductLayer   ",
"TIDL_SoftMaxLayer        ",
"TIDL_BatchNormLayer      ",
"TIDL_BiasLayer           ",
"TIDL_ScaleLayer          ",
"TIDL_Deconv2DLayer       ",
"TIDL_ConcatLayer         ",
"TIDL_SplitLayer          ",
"TIDL_SliceLayer          ",
"TIDL_CropLayer           ",
"TIDL_FlatternLayer       ",
"TIDL_DropOutLayer        ",
"TIDL_ArgMaxLayer         ",
};

void setDefaultParams(tidl_import_config * params)
{
  params->randParams          = 0;
  params->modelType           = 0; // 0 - caffe, 1- tensorFlow
  params->quantizationStyle   = TIDL_quantStyleDynamic;
  params->quantRoundAdd       = 50; // 0 - caffe, 1- tensorFlow
  params->numParamBits        = 8;
  params->rawSampleInData     = 0; // 0 - Encoded, 1- RAW
  params->numSampleInData     = 1;
  params->foldBnInConv2D      = 1;
  params->preProcType         = 0;
  params->conv2dKernelType    = 0;
  params->inElementType       = TIDL_SignedChar;
  params->inQuantFactor       = -1;

  params->inWidth               = -1;
  params->inHeight              = -1;
  params->inNumChannels         = -1;

}

void tidlQuantStatsTool(tidl_import_config * params)
{
  FILE * fp;
  char sysCommand[500];

  sprintf(sysCommand, "if exist tempDir rmdir /S/Q tempDir");
  system(sysCommand);
  sprintf(sysCommand, "mkdir tempDir");
  system(sysCommand);


  fp = fopen("tempDir\\configFilesList.txt", "w+");
  if(fp== NULL)
  {
    printf("Could not open config  file tempDir\\configFilesList.txt  \n");
    return;
  }
  fprintf(fp, "1 .\\tempDir\\qunat_stats_config.txt \n0\n" );
  fclose(fp);

  fp = fopen("tempDir\\qunat_stats_config.txt", "w+");
  if(fp== NULL)
  {
    printf("Could not open config  file tempDir\\qunat_stats_config.txt  \n");
    return;
  }
  fprintf(fp, "rawImage    = %d\n",params->rawSampleInData);
  fprintf(fp, "numFrames   = %d\n",params->numSampleInData);
  fprintf(fp, "preProcType  = %d\n",params->preProcType);
  fprintf(fp, "inData   = %s\n",params->sampleInData);
  fprintf(fp, "traceDumpBaseName   = \".\\tempDir\\trace_dump_\"\n");
  fprintf(fp, "outData   = \".\\tempDir\\stats_tool_out.bin\"\n");
  fprintf(fp, "updateNetWithStats   = 1\n");
  fprintf(fp, "outputNetBinFile     = %s\n",params->outputNetFile);
  fprintf(fp, "paramsBinFile        = %s\n",params->outputParamsFile);
  fprintf(fp, "netBinFile   = \".\\tempDir\\temp_net.bin\"\n");
  fclose(fp);

  sprintf(sysCommand, "copy  %s .\\tempDir\\temp_net.bin",params->outputNetFile) ;
  system(sysCommand);
  // printf("Quantstat checkpoint(0)\n");
  // return;
  sprintf(sysCommand, " %s .\\tempDir\\configFilesList.txt",params->tidlStatsTool);
  printf(sysCommand);
  system(sysCommand);

  return;

}

/**
----------------------------------------------------------------------------
@ingroup    TIDL_Import
@fn         tidlValidateImportParams
@brief      Function validates input parameters related to tidl import
            sets appropriate error in response to violation from
            expected values.

@param      params : TIDL Create time parameters
@remarks    None
@return     Error related to parameter.
----------------------------------------------------------------------------
*/
int32_t tidlValidateImportParams(tidl_import_config * params)
{

  /* randParams can be either 0 or 1*/
  if((params->randParams != 0) && (params->randParams != 1))
  {
    printf("\n Invalid randParams setting : set either 0 or 1");
    return -1;
  }
  /* modelType can be either 0 or 1*/
  else if((params->modelType != 0) && (params->modelType != 1))
  {
    printf("\n Invalid modelType parameter setting : set either 0 or 1");
    return -1;
  }
  /* Currently quantizationStyle = 1 is supported */
  else if(params->quantizationStyle != 1)
  {
    printf("\n Invalid quantizationStyle parameter setting : set it to 1");
    return -1;
  }
  /* quantRoundAdd can be 0 to 100 */
  else if((params->quantRoundAdd < 0) || (params->quantRoundAdd > 100))
  {
    printf("\n Invalid quantRoundAdd parameter setting : set it 0 to 100");
    return -1;
  }
  /* numParamBits can be 4 to 12 */
  else if((params->numParamBits < 4) || (params->numParamBits > 12))
  {
    printf("\n Invalid numParamBits parameter setting : set it 4 to 12");
    return -1;
  }
  /* rawSampleInData can be either 0 or 1*/
  else if((params->rawSampleInData != 0) && (params->rawSampleInData != 1))
  {
    printf("\n Invalid rawSampleInData parameter setting : set either 0 or 1");
    return -1;
  }
  /* numSampleInData can be >0  */
  else if(params->numSampleInData <= 0)
  {
    printf("\n Invalid numSampleInData parameter setting : set it to >0 ");
    return -1;
  }
  /* foldBnInConv2D can be either 0 or 1*/
  else if((params->foldBnInConv2D != 0) && (params->foldBnInConv2D != 1))
  {
    printf("\n Invalid foldBnInConv2D parameter setting : set either 0 or 1");
    return -1;
  }
  /* preProcType can be 0 to 3 */
  else if((params->preProcType < 0) || (params->preProcType > 3))
  {
    printf("\n Invalid preProcType parameter setting : set it 0 to 3");
    return -1;
  }
  /* conv2dKernelType can be either 0 or 1*/
  else if((params->conv2dKernelType != 0) && (params->conv2dKernelType != 1))
  {
    printf("\n Invalid conv2dKernelType parameter setting : set either 0 or 1");
    return -1;
  }
  /* inElementType can be either 0 or 1*/
  else if((params->inElementType != 0) && (params->inElementType != 1))
  {
    printf("\n Invalid inElementType parameter setting : set either 0 or 1");
    return -1;
  }
  /* inQuantFactor can be >0  */
  else if((params->inQuantFactor < -1) || (params->inQuantFactor == 0))
  {
    printf("\n Invalid inQuantFactor parameter setting : set it to >0 ");
    return -1;
  }
  /* inWidth can be >0  */
  else if((params->inWidth < -1) || (params->inWidth == 0))
  {
    printf("\n Invalid inWidth parameter setting : set it to >0 ");
    return -1;
  }
  /* inHeight can be >0  */
  else if((params->inHeight < -1) || (params->inHeight == 0))
  {
    printf("\n Invalid inHeight parameter setting : set it to >0 ");
    return -1;
  }
  /* inNumChannels can be 1 to 1024  */
  else if((params->inNumChannels < -1) || (params->inNumChannels == 0) || (params->inNumChannels > 1024))
  {
    printf("\n Invalid inNumChannels parameter setting : set it 1 to 1024 ");
    return -1;
  }
  else
  {
    return 0;
  }
}

//for output caffemodel file only
char *gpModeltxtFile;
int giopmodeltxtEn;

sTIDL_OrgNetwork_t      orgTIDLNetStructure;
sTIDL_Network_t         tIDLNetStructure;
void caffe_import( tidl_import_config * params);
void tf_import(tidl_import_config * params);

int32_t main(int32_t argc, char *argv[])
{
  int32_t status = 0;
  int32_t runQuantStep = 0;
  FILE * fp;



  if(argc < 2)
  {
    printf("Number of input parameters are not enough \n");
    printf("Usage : \n tidl_model_import.out.exe config_file.txt\n");
    exit(-1);
  }

  if(argc > 2)
  {
    gpModeltxtFile = argv[2];
    giopmodeltxtEn = 1;
    printf("Will Output the caffemodel file as txt file %s later...\n",gpModeltxtFile );
  }
  else
  {
    giopmodeltxtEn = 0;
  }

  fp = fopen(argv[1], "r");
  if(fp== NULL)
  {
    printf("Could not open config  file : %s  \n",argv[1]);
    return(0);
  }
  fclose(fp);

  setDefaultParams(&gParams) ;

  status = readparamfile(argv[1], &gsTokenMap_tidl_import_config[0]) ;

  if(status == -1)
  {
    printf("Parser Failed");
    return -1 ;
  }
  //debug
    printf("Reading config file %s\n",argv[1]);
  //debug
  status = tidlValidateImportParams(&gParams);
  if(status == -1)
  {
    printf("\n Validation of Parameters Failed \n");
    return -1 ;
  }

  /*  inputNetFile && inputParamsFile */
  fp = fopen((const char *)gParams.inputNetFile, "r");
  if(fp== NULL)
  {
    printf("Couldn't open inputNetFile file: %s  \n", gParams.inputNetFile);
    return(0);
  }
  fclose(fp);

  if(gParams.modelType == 0)
  {
    fp = fopen((const char *)gParams.inputParamsFile, "r");
    if(fp== NULL)
    {
      printf("Couldn't open inputParamsFile file: %s  \n", gParams.inputParamsFile);
      return(0);
    }
    fclose(fp);
  }


  if(gParams.modelType == 0)
  {
    if(gParams.inQuantFactor == -1)
    {
      gParams.inQuantFactor = 255;
    }
    //
    printf("Begining importing caffe model...\n");
    //
    caffe_import(&gParams);
  }
  else if (gParams.modelType == 1)
  {
    if(gParams.inQuantFactor == -1)
    {
      gParams.inQuantFactor = 128*255;
    }
    tf_import(&gParams);
  }



  fp = fopen((const char *)gParams.sampleInData, "r");
  if(fp== NULL)
  {
    printf("Couldn't open sampleInData file: %s  \n", gParams.sampleInData);
    return(0);
  }
  fclose(fp);

  fp = fopen((const char *)gParams.tidlStatsTool, "r");
  if(fp== NULL)
  {
    printf("Couldn't open tidlStatsTool file: %s  \n", gParams.tidlStatsTool);
    return(0);
  }
  fclose(fp);


  tidlQuantStatsTool(&gParams);


  return (0);
}
