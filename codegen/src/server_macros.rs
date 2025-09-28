use proc_macro2::TokenStream;
use quote::{ToTokens, TokenStreamExt as _, format_ident, quote};
use syn::Type;

use crate::hook_fn::ParamKind::*;
use crate::hook_fn::{HookFn, InputHandleOp, Param};
use crate::utils::{result_ident, usize_from};

pub fn exe(input: &HookFn) -> TokenStream {
    let params = input.params.as_ref();

    let recv_statements = params.iter().map(|param| {
        let Param { name, name_str, .. } = param;
        let ptr_ident = param.exe_ptr();
        // TODO: can reference into the buffer if aligned
        match &param.kind {
            InputValue => {
                let ty = param.raw_type();
                quote! {
                    let #name: #ty = recv_ctx.recv(#name_str);
                }
            }
            InputHandle(op) => {
                let ty = param.raw_type();
                let is_destroy = match op {
                    InputHandleOp::Use | InputHandleOp::Modify => false,
                    InputHandleOp::Destroy => true,
                };
                let set_handle_proxy = if let InputHandleOp::Modify = op {
                    quote! { handle_proxy = Some(#name as usize); }
                } else {
                    Default::default()
                };
                quote! {
                    let #name: #ty = recv_ctx.recv(#name_str);
                    let #name = if server.opt_shadow_desc {
                        #set_handle_proxy
                        let is_destroy = #is_destroy;
                        server.resources.get(#name, is_destroy)
                    } else {
                        #name
                    };
                }
            }
            InputSinglePtr => {
                let ty = param.deref_type();
                quote! {
                    let #name: #ty = recv_ctx.recv(#name_str);
                    let #ptr_ident = (&raw const #name).cast_mut();
                }
            }
            InputArrayPtr { is_void_ptr, .. } => {
                let ty = use_u8_if_void(param.deref_type(), *is_void_ptr);
                quote! {
                    let #name = recv_ctx.recv_slice::<#ty>(#name_str);
                    let #ptr_ident = #name.as_ptr().cast_mut();
                }
            }
            InputCStr => {
                quote! {
                    let #name = recv_ctx.recv_cstr(#name_str);
                    let #ptr_ident = #name.as_ptr();
                }
            }
            OutputHandle => {
                quote! {
                    if server.opt_shadow_desc {
                        handle_proxy = Some(recv_ctx.recv(#name_str));
                    }
                }
            }
            OutputSinglePtr | OutputArrayPtr { .. } => Default::default(),
            DeviceInputPtr | DeviceOutputPtr => {
                let ty = param.raw_type();
                quote! {
                    let #name: #ty = recv_ctx.recv(#name_str);
                }
            }
            Skip | Const(_) => Default::default(),
        }
    });

    let def_statements = params.iter().filter(|p| p.is_host_output()).map(|param| {
        let Param { name, .. } = param;
        let ptr_ident = param.exe_ptr();
        match &param.kind {
            OutputHandle | OutputSinglePtr => {
                let ty = param.deref_type();
                let set_handle_output = if let OutputHandle = param.kind {
                    quote! {
                        if server.opt_shadow_desc {
                            handle_output = Some(#ptr_ident.cast());
                        }
                    }
                } else {
                    Default::default()
                };
                quote! {
                    let mut #name = std::mem::MaybeUninit::<#ty>::uninit();
                    let #ptr_ident = #name.as_mut_ptr();
                    #set_handle_output
                }
            }
            OutputArrayPtr { is_void_ptr, len, cap } => {
                let ty = use_u8_if_void(param.deref_type(), *is_void_ptr);
                let cap = usize_from(cap.as_ref().unwrap_or(len));
                quote! {
                    let mut #name = Box::<[#ty]>::new_uninit_slice(#cap);
                    let #ptr_ident = std::mem::MaybeUninit::slice_as_mut_ptr(&mut #name);
                }
            }
            kind => panic!("unhandled kind: {kind:?}"),
        }
    });

    let assume_init = params.iter().filter(|p| p.is_host_output()).map(|param| {
        let Param { name, .. } = param;
        quote! {
            let #name = unsafe { #name.assume_init() };
        }
    });

    // execution statement
    let result_name = result_ident();
    let exec_statement = if !input.injections.server_execution.is_empty() {
        let mut tokens = TokenStream::new();
        tokens.append_all(&input.injections.server_execution);
        tokens
    } else {
        let result_ty = input.return_type();
        let exec_args = params.iter().filter_map(|param| {
            let ptr_ident = param.exe_ptr();
            match &param.kind {
                InputValue | InputHandle(_) | DeviceInputPtr | DeviceOutputPtr => {
                    Some(param.name.to_token_stream())
                }
                InputSinglePtr if param.is_hacked_type() => Some(quote!(#ptr_ident.cast())),
                InputSinglePtr | InputCStr | OutputHandle | OutputSinglePtr => {
                    Some(quote!(#ptr_ident))
                }
                InputArrayPtr { is_void_ptr, .. } | OutputArrayPtr { is_void_ptr, .. } => {
                    if *is_void_ptr {
                        Some(quote!(#ptr_ident.cast()))
                    } else {
                        Some(quote!(#ptr_ident))
                    }
                }
                Skip => None,
                Const(expr) => Some(quote!(#expr)),
            }
        });
        let func = input.parent.as_ref().unwrap_or(input.func());
        quote! {
            let #result_name: #result_ty = unsafe { #func(#(#exec_args),*) };
        }
    };

    let send_statements = params.iter().filter(|p| p.is_host_output()).map(|param| {
        let Param { name, name_str, .. } = param;
        match &param.kind {
            OutputHandle | OutputSinglePtr => quote! {
                send_ctx.send(&#name, #name_str);
            },
            OutputArrayPtr { cap: None, .. } => {
                quote! {
                    send_ctx.send_slice(&#name, #name_str);
                }
            }
            OutputArrayPtr { len, .. } => {
                let len = usize_from(len);
                quote! {
                    send_ctx.send_slice(&#name[..#len], #name_str);
                }
            }
            kind => panic!("unhandled kind: {kind:?}"),
        }
    });

    let is_create_handle = input.is_create_handle();
    let is_modify_handle = input.is_modify_handle();
    let is_async_api = input.is_async_api();

    let server_extra_recv = input.injections.server_extra_recv.iter();
    let server_before_execution = input.injections.server_before_execution.iter();
    let server_after_send = input.injections.server_after_send.iter();

    let proc_id = &input.proc_id;
    let func = input.func();
    let func_str = func.to_string();
    let func_exe = format_ident!("{func}Exe");

    quote! {
        pub fn #func_exe(server: &mut ServerWorker) {
            server.before_call(#func_str);

            // set if opt_shadow_desc && (is_create_handle || is_modify_handle)
            let mut handle_proxy: Option<usize> = None;
            // set if opt_shadow_desc && is_create_handle
            let mut handle_output: Option<*mut usize> = None;

            let is_phos = cfg!(feature = "phos") && server.opt_shadow_desc;
            let is_create_handle = #is_create_handle;
            let is_modify_handle = #is_modify_handle;

            let mut recv_ctx = network::session::RecvSession::begin_server(
                server.id,
                &server.channel_receiver,
                #func_str,
                is_phos && (is_create_handle || is_modify_handle),
                #proc_id,
            );
            #( #recv_statements )*

            #( #server_extra_recv )*
            let save_args = recv_ctx.finish();
            #( #def_statements )*
            #( #server_before_execution )*
            #exec_statement
            #( #assume_init )*

            if #result_name.is_error() {
                log::error!(target: #func_str, "[#{}] returned error: {:?}", server.id, #result_name);
            }
            if is_create_handle && server.opt_shadow_desc {
                server.resources.insert(handle_proxy.unwrap(), unsafe { *handle_output.unwrap() });
            }
            #[cfg(feature = "phos")]
            if let Some(save_args) = save_args {
                server.resources.insert_args(handle_proxy.unwrap(), save_args);
            }
            if is_create_handle && server.opt_shadow_desc {
                return;
            }

            let is_async_api = #is_async_api;
            if is_async_api && server.opt_async_api {
                return;
            }

            let send_ctx = network::session::SendSession::begin(
                server.id,
                &server.channel_sender,
                #func_str,
            );

            #( #send_statements )*
            send_ctx.send(&#result_name, stringify!(#result_name));
            send_ctx.finish();
            #( #server_after_send )*
        }
    }
}

fn use_u8_if_void(ty: &Type, is_void_ptr: bool) -> TokenStream {
    if is_void_ptr { quote!(u8) } else { quote!(#ty) }
}
