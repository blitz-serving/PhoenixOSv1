use crate::types::cublasLt::*;
use codegen::cuda_hook;
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_hook(proc_id = 1500)]
fn cublasLtCreate(#[handle = "create"] lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1501)]
fn cublasLtDestroy(#[handle = "destroy"] lightHandle: cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1511)]
fn cublasLtMatmul(
    #[handle = "use"] lightHandle: cublasLtHandle_t,
    #[handle = "use"] computeDesc: cublasLtMatmulDesc_t,
    #[host] alpha: *const HackedAssumeDouble, // FIXME: safe until we support setting pointer mode
    #[device] A: *const c_void,
    #[handle = "use"] Adesc: cublasLtMatrixLayout_t,
    #[device] B: *const c_void,
    #[handle = "use"] Bdesc: cublasLtMatrixLayout_t,
    #[host] beta: *const HackedAssumeDouble,
    #[device] C: *const c_void,
    #[handle = "use"] Cdesc: cublasLtMatrixLayout_t,
    #[device] D: *mut c_void,
    #[handle = "use"] Ddesc: cublasLtMatrixLayout_t,
    #[host] algo: *const cublasLtMatmulAlgo_t, // FIXME: nullable
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
    #[handle = "use"] stream: cudaStream_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1516)]
fn cublasLtMatmulAlgoGetHeuristic(
    #[handle = "use"] lightHandle: cublasLtHandle_t,
    #[handle = "use"] operationDesc: cublasLtMatmulDesc_t,
    #[handle = "use"] Adesc: cublasLtMatrixLayout_t,
    #[handle = "use"] Bdesc: cublasLtMatrixLayout_t,
    #[handle = "use"] Cdesc: cublasLtMatrixLayout_t,
    #[handle = "use"] Ddesc: cublasLtMatrixLayout_t,
    #[handle = "use"] preference: cublasLtMatmulPreference_t,
    requestedAlgoCount: c_int,
    #[host(output, len = requestedAlgoCount)]
    heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
    returnAlgoCount: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1519)]
fn cublasLtMatmulDescCreate(
    #[handle = "create"] matmulDesc: *mut cublasLtMatmulDesc_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1521)]
fn cublasLtMatmulDescDestroy(
    #[handle = "destroy"] matmulDesc: cublasLtMatmulDesc_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1523)]
fn cublasLtMatmulDescSetAttribute(
    #[handle = "modify"] matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t {
    'client_before_send: {
        assert_ne!(attr, cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE);
    }
}

#[cuda_hook(proc_id = 1524)]
fn cublasLtMatmulPreferenceCreate(
    #[handle = "create"] pref: *mut cublasLtMatmulPreference_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1526)]
fn cublasLtMatmulPreferenceDestroy(
    #[handle = "destroy"] pref: cublasLtMatmulPreference_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1528)]
fn cublasLtMatmulPreferenceSetAttribute(
    #[handle = "modify"] pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1529)]
fn cublasLtMatrixLayoutCreate(
    #[handle = "create"] matLayout: *mut cublasLtMatrixLayout_t,
    type_: cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1531)]
fn cublasLtMatrixLayoutDestroy(
    #[handle = "destroy"] matLayout: cublasLtMatrixLayout_t,
) -> cublasStatus_t;
