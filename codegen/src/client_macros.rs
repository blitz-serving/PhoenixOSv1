use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::hook_fn::ParamKind::*;
use crate::hook_fn::{HookFn, Param};
use crate::utils::{result_ident, usize_from};

pub fn hijack(input: &HookFn) -> TokenStream {
    let params = input.params.as_ref();

    let send_statements = params.iter().map(|param| {
        let Param { name, name_str, .. } = param;
        match &param.kind {
            InputValue | InputHandle { .. } | DeviceInputPtr | DeviceOutputPtr => {
                quote! { send_ctx.send(&#name, #name_str); }
            }
            InputSinglePtr if param.is_hacked_type() => {
                quote! { unsafe { send_ctx.send_unaligned(#name, #name_str) }; }
            }
            InputSinglePtr => {
                quote! {
                    send_ctx.check_not_null(#name, #name_str);
                    send_ctx.send(unsafe { &*#name }, #name_str);
                }
            }
            InputArrayPtr { is_void_ptr, len } => {
                let ptr = cast_void_to_u8(name, *is_void_ptr);
                let len = usize_from(len);
                quote! {
                    let #name = unsafe { send_ctx.slice_from(#ptr, #len, #name_str) };
                    send_ctx.send_slice(#name, #name_str);
                }
            }
            InputCStr => {
                quote! {
                    let #name = unsafe { std::ffi::CStr::from_ptr(#name) };
                    send_ctx.send_cstr(#name, #name_str);
                }
            }
            OutputHandle { .. } => {
                quote! {
                    if client.opt_shadow_desc {
                        send_ctx.send_handle(#name, #name_str, next_handle);
                    }
                }
            }
            OutputSinglePtr | OutputArrayPtr { .. } | Skip => Default::default(),
            Const(expr) => {
                quote! { assert_eq!(#name, #expr); }
            }
        }
    });

    let recv_statements = params.iter().filter(|p| p.is_host_output()).map(|param| {
        let Param { name, name_str, .. } = param;
        match &param.kind {
            OutputHandle { .. } | OutputSinglePtr => {
                quote! {
                    // FIXME: allocate space for null pointers
                    let #name = unsafe { recv_ctx.mut_from(#name, #name_str) };
                    recv_ctx.recv_mut(#name, #name_str);
                }
            }
            OutputArrayPtr { is_void_ptr, len, .. } => {
                let ptr = cast_void_to_u8(name, *is_void_ptr);
                let len = usize_from(len);
                quote! {
                    let #name = unsafe { recv_ctx.mut_slice_from(#ptr, #len, #name_str) };
                    recv_ctx.recv_mut_slice(#name, #name_str);
                }
            }
            kind => panic!("unhandled kind: {kind:?}"),
        }
    });

    let params = params.iter().map(|param| {
        let name = &param.name;
        let ty = param.raw_type();
        quote! { #name: #ty }
    });
    let result_name = result_ident();
    let result_ty = input.return_type();

    let is_create_handle = input.is_create_handle();
    let is_async_api = input.is_async_api();

    let client_before_send = input.injections.client_before_send.iter();
    let client_extra_send = input.injections.client_extra_send.iter();
    let client_initial_recv = input.injections.client_initial_recv.iter();
    let client_after_recv = input.injections.client_after_recv.iter();

    let modifiers = match input.parent {
        Some(_) => quote!(pub),
        None => quote!(#[unsafe(no_mangle)] pub extern "C"),
    };

    let proc_id = &input.proc_id;
    let func = input.func();
    let func_str = func.to_string();

    quote! {
        #modifiers fn #func(#(#params),*) -> #result_ty {
            let main = |client: &mut ClientThread| {
                let proc_id: i32 = #proc_id;

                #( #client_before_send )*

                let mut send_ctx = network::session::SendSession::begin(
                    client.id,
                    &mut client.channel_sender,
                    #func_str,
                );

                send_ctx.send(&proc_id, "proc_id");
                #( #send_statements )*

                #( #client_extra_send )*

                send_ctx.finish();

                let is_create_handle = #is_create_handle;
                if is_create_handle && client.opt_shadow_desc {
                    return Default::default();
                }

                let is_async_api = #is_async_api;
                if is_async_api && client.opt_async_api {
                    return Default::default();
                }

                let mut recv_ctx = network::session::RecvSession::begin(
                    client.id,
                    &mut client.channel_receiver,
                    #func_str,
                );

                #( #client_initial_recv )*
                #( #recv_statements )*
                let #result_name: #result_ty = recv_ctx.recv(stringify!(#result_name));
                recv_ctx.finish();
                if cudasys::types::CheckError::is_error(#result_name) {
                    log::error!(
                        target: #func_str,
                        "[#{}] returned error: {:?}\n{}",
                        client.id,
                        #result_name,
                        std::backtrace::Backtrace::force_capture(),
                    );
                }
                #( #client_after_recv )*

                return #result_name;
            };
            ClientThread::with_borrow_mut(|client| {
                client.before_call(#func_str);
                let result = main(client);
                client.after_call();
                result
            })
        }
    }
}

fn cast_void_to_u8(name: &Ident, is_void_ptr: bool) -> TokenStream {
    if is_void_ptr {
        quote!(#name.cast::<u8>())
    } else {
        quote!(#name)
    }
}
