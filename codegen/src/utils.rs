use hookdef::last_seg;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::spanned::Spanned as _;
use syn::{Expr, Type, TypePtr};

pub fn is_handle_type_with_op_key(ty: &Type) -> bool {
    let Some(ident) = last_seg(ty) else { return false };
    matches!(
        ident.to_string().as_str(),
        "cublasHandle_t"
            | "CUfunction"
            | "cublasLtMatmulDesc_t"
            | "cublasLtMatmulPreference_t"
            | "cudaEvent_t"
    )
}

pub fn is_handle_type(ty: &Type) -> bool {
    let Some(ident) = last_seg(ty) else { return false };
    matches!(
        ident.to_string().as_str(),
        "CUstream"
            // | "CUcontext"
            // | "CUmodule"
            | "CUfunction"
            | "cudaStream_t"
            | "cudaEvent_t"
            // | "cudaIpcMemHandle_t"
            | "cudnnHandle_t"
            | "cudnnActivationDescriptor_t"
            | "cudnnConvolutionDescriptor_t"
            | "cudnnFilterDescriptor_t"
            | "cudnnTensorDescriptor_t"
            | "cudnnBackendDescriptor_t"
            | "cublasHandle_t"
            | "cublasLtHandle_t"
            | "cublasLtMatmulDesc_t"
            | "cublasLtMatrixLayout_t"
            | "cublasLtMatmulPreference_t"
    )
}

pub fn is_async_return_type(ty: &Type) -> bool {
    let Some(ident) = last_seg(ty) else { return false };
    matches!(
        ident.to_string().as_str(),
        "CUresult"
            | "cudaError_t"
            | "nvmlReturn_t"
            | "cudnnStatus_t"
            | "cublasStatus_t"
            | "ncclResult_t"
    )
}

pub fn is_void_ptr(ptr: &TypePtr) -> bool {
    last_seg(&ptr.elem).is_some_and(|seg| seg == "c_void")
}

pub fn is_const_cstr(ptr: &TypePtr) -> bool {
    ptr.const_token.is_some() && last_seg(&ptr.elem).is_some_and(|seg| seg == "c_char")
}

pub fn usize_from(expr: &Expr) -> TokenStream {
    let try_into = quote_spanned!(expr.span()=> try_into);
    quote!({
        #[allow(clippy::useless_conversion)]
        let i: usize = (#expr).to_owned().#try_into().unwrap();
        i
    })
}

pub fn result_ident() -> Ident {
    Ident::new("hook_result", Span::call_site())
}
