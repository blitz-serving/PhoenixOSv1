#![feature(proc_macro_diagnostic)]

mod client_macros;
mod hook_fn;
mod server_macros;
mod utils;

use hookdef::{CustomHookAttrs, CustomHookFn};
use proc_macro::TokenStream;
use syn::parse_macro_input;

use crate::hook_fn::HookFn;

/// Basic checks on a hook declaration
#[proc_macro_attribute]
pub fn cuda_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func.into_plain_fn().into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn cuda_custom_hook(args: TokenStream, input: TokenStream) -> TokenStream {
    if let Err(err) = CustomHookAttrs::from_macro(args.into()) {
        return err.to_compile_error().into();
    }
    parse_macro_input!(input as CustomHookFn).to_plain_fn().into()
}

/// The procedural macro to generate hijack functions for client intercepting application calls.
#[proc_macro_attribute]
pub fn cuda_hook_hijack(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    client_macros::hijack(&input).into()
}

/// The procedural macro to generate execution functions for the server dispatcher.
#[proc_macro_attribute]
pub fn cuda_hook_exe(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut input = match HookFn::parse(args.into(), input.into()) {
        Ok(func) => func,
        Err(err) => return err.to_compile_error().into(),
    };

    for param in input.params.iter_mut() {
        param.setup_exe_ptr();
    }

    server_macros::exe(&input).into()
}
