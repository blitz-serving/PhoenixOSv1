use crate::types::cudnn::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_hook(proc_id = 1804)]
fn cudnnCreate(#[handle = "create"] handle: *mut cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1814)]
fn cudnnCreateTensorDescriptor(
    #[handle = "create"] tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1873)]
fn cudnnSetTensor4dDescriptor(
    #[handle = "modify"] tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1805)]
fn cudnnCreateActivationDescriptor(
    #[handle = "create"] activationDesc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1861)]
fn cudnnSetActivationDescriptor(
    #[handle = "modify"] activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1816)]
fn cudnnDestroy(#[handle = "destroy"] handle: cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2126)]
fn cudnnSetConvolution2dDescriptor(
    #[handle = "modify"] convDesc: cudnnConvolutionDescriptor_t,
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1871, async_api)]
fn cudnnSetStream(
    #[handle = "modify"] handle: cudnnHandle_t,
    #[handle = "use"] streamId: cudaStream_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1875, async_api)]
fn cudnnSetTensorNdDescriptor(
    #[handle = "modify"] tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: c_int,
    #[host(len = nbDims)] dimA: *const c_int,
    #[host(len = nbDims)] strideA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1826, async_api)]
fn cudnnDestroyTensorDescriptor(
    #[handle = "destroy"] tensorDesc: cudnnTensorDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1808)]
fn cudnnCreateFilterDescriptor(
    #[handle = "create"] filterDesc: *mut cudnnFilterDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1820, async_api)]
fn cudnnDestroyFilterDescriptor(
    #[handle = "destroy"] filterDesc: cudnnFilterDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1865, async_api)]
fn cudnnSetFilterNdDescriptor(
    #[handle = "modify"] filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: c_int,
    #[host(len = nbDims)] filterDimA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2104)]
fn cudnnCreateConvolutionDescriptor(
    #[handle = "create"] convDesc: *mut cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2105, async_api)]
fn cudnnDestroyConvolutionDescriptor(
    #[handle = "destroy"] convDesc: cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2129, async_api)]
fn cudnnSetConvolutionNdDescriptor(
    #[handle = "modify"] convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: c_int,
    #[host(len = arrayLength)] padA: *const c_int,
    #[host(len = arrayLength)] filterStrideA: *const c_int,
    #[host(len = arrayLength)] dilationA: *const c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2127, async_api)]
fn cudnnSetConvolutionGroupCount(
    #[handle = "modify"] convDesc: cudnnConvolutionDescriptor_t,
    groupCount: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2128, async_api)]
fn cudnnSetConvolutionMathType(
    #[handle = "modify"] convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2130)]
fn cudnnSetConvolutionReorderType(
    #[handle = "modify"] convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2116)]
fn cudnnGetConvolutionForwardAlgorithm_v7(
    #[handle = "use"] handle: cudnnHandle_t,
    #[handle = "use"] srcDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] filterDesc: cudnnFilterDescriptor_t,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    #[handle = "use"] destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2103, async_api)]
fn cudnnConvolutionForward(
    #[handle = "use"] handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    #[handle = "use"] wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2008)]
fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] zDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2009)]
fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[handle = "use"] activationDesc: cudnnActivationDescriptor_t,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2004, async_api)]
fn cudnnBatchNormalizationForwardTrainingEx(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    #[handle = "use"] zDesc: cudnnTensorDescriptor_t,
    #[device] zData: *const c_void,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *mut c_void,
    #[handle = "use"] bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    exponentialAverageFactor: f64,
    #[device] resultRunningMean: *mut c_void,
    #[device] resultRunningVariance: *mut c_void,
    epsilon: f64,
    #[device] resultSaveMean: *mut c_void,
    #[device] resultSaveInvVariance: *mut c_void,
    #[handle = "use"] activationDesc: cudnnActivationDescriptor_t,
    #[device] workspace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2007)]
fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] dyDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] dzDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] dxDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2002, async_api)]
fn cudnnBatchNormalizationBackwardEx(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alphaDataDiff: *const HackedAssumeDouble,
    #[host] betaDataDiff: *const HackedAssumeDouble,
    #[host] alphaParamDiff: *const HackedAssumeDouble,
    #[host] betaParamDiff: *const HackedAssumeDouble,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *const c_void,
    #[handle = "use"] dyDesc: cudnnTensorDescriptor_t,
    #[device] dyData: *const c_void,
    #[handle = "use"] dzDesc: cudnnTensorDescriptor_t,
    #[device] dzData: *mut c_void,
    #[handle = "use"] dxDesc: cudnnTensorDescriptor_t,
    #[device] dxData: *mut c_void,
    #[handle = "use"] dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    #[device] bnScaleData: *const c_void,
    #[device] bnBiasData: *const c_void,
    #[device] dBnScaleData: *mut c_void,
    #[device] dBnBiasData: *mut c_void,
    epsilon: f64,
    #[device] savedMean: *const c_void,
    #[device] savedInvVariance: *const c_void,
    #[handle = "use"] activationDesc: cudnnActivationDescriptor_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2113)]
fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    #[handle = "use"] handle: cudnnHandle_t,
    #[handle = "use"] filterDesc: cudnnFilterDescriptor_t,
    #[handle = "use"] diffDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    #[handle = "use"] gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2101, async_api)]
fn cudnnConvolutionBackwardData(
    #[handle = "use"] handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[handle = "use"] wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    #[handle = "use"] dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    #[handle = "use"] dxDesc: cudnnTensorDescriptor_t,
    #[device] dx: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2213)]
fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    #[handle = "use"] handle: cudnnHandle_t,
    #[handle = "use"] srcDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] diffDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    #[handle = "use"] gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2202, async_api)]
fn cudnnConvolutionBackwardFilter(
    #[handle = "use"] handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    #[handle = "use"] dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    #[handle = "use"] dwDesc: cudnnFilterDescriptor_t,
    #[device] dw: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1802, async_api)]
fn cudnnBatchNormalizationForwardInference(
    #[handle = "use"] handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
    #[handle = "use"] bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    #[device] estimatedMean: *const c_void,
    #[device] estimatedVariance: *const c_void,
    epsilon: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1864)]
fn cudnnSetFilter4dDescriptor(
    #[handle = "modify"] filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2121)]
fn cudnnGetConvolutionNdForwardOutputDim(
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    #[handle = "use"] inputTensorDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] filterDesc: cudnnFilterDescriptor_t,
    nbDims: c_int,
    #[host(output, len = nbDims)] tensorOuputDimA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2117)]
fn cudnnGetConvolutionForwardWorkspaceSize(
    #[handle = "use"] handle: cudnnHandle_t,
    #[handle = "use"] xDesc: cudnnTensorDescriptor_t,
    #[handle = "use"] wDesc: cudnnFilterDescriptor_t,
    #[handle = "use"] convDesc: cudnnConvolutionDescriptor_t,
    #[handle = "use"] yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_custom_hook] // local
fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

#[cuda_hook(proc_id = 2500)]
fn cudnnBackendCreateDescriptor(
    descriptorType: cudnnBackendDescriptorType_t,
    #[handle = "create"] descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2501, async_api)]
fn cudnnBackendDestroyDescriptor(
    #[handle = "destroy"] descriptor: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2502, async_api)]
fn cudnnBackendExecute(
    #[handle = "use"] handle: cudnnHandle_t,
    #[handle = "use"] executionPlan: cudnnBackendDescriptor_t,
    #[handle = "use"] variantPack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2503)]
fn cudnnBackendFinalize(#[handle = "modify"] descriptor: cudnnBackendDescriptor_t)
-> cudnnStatus_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992504, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeCount(
    #[handle = "use"] descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[value = std::ptr::null_mut()] arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992505, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeData(
    #[handle = "use"] descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // not CUDNN_TYPE_BACKEND_DESCRIPTOR
    requestedElementCount: i64,
    elementCount: *mut i64,
    // `len` and `cap` are required because some callers provide buffers shorter than `requestedElementCount`
    // so that using `len = requestedElementCount` alone will lead to memory corruption
    // https://github.com/NVIDIA/cudnn-frontend/blob/v1.11.0/include/cudnn_frontend_ExecutionPlan.h#L165-L177
    #[host(
        output,
        len = attributeType.data_size() * elementCount.to_owned(),
        cap = attributeType.data_size() * requestedElementCount,
    )]
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

// FIXME: this actually "modifies" all descriptors in arrayOfElements
#[cuda_hook(proc_id = 992506, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeDescriptors(
    #[handle = "use"] descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    #[value = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR]
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[host(input, len = attributeType.data_size() * requestedElementCount)]
    arrayOfElements: *mut c_void, // FIXME: may be output, but nobody use it that way
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2506, async_api)]
fn cudnnBackendSetAttribute(
    #[handle = "modify"] descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // TODO: might be CUDNN_TYPE_BACKEND_DESCRIPTOR
    elementCount: i64,
    #[host(len = attributeType.data_size() * elementCount)] arrayOfElements: *const c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1850)]
fn cudnnGetVersion() -> usize;
