//! Semantic parsing of hook definitions.

use std::borrow::Cow;
use std::mem;

use hookdef::{HookAttrs, HookFnItem, HookInjections, check_max_attributes, is_hacked_type};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{
    Attribute, Error, Expr, FnArg, Lit, LitInt, Meta, Pat, PatIdent, PatType, Result, ReturnType,
    Signature, Token, Type, TypePtr, parse_quote,
};

use crate::utils::{is_async_return_type, is_const_cstr, is_handle_type, is_void_ptr};

pub struct HookFn {
    pub proc_id: LitInt,
    is_async_api: Option<bool>,
    pub parent: Option<Ident>,
    min_cuda_version: u8,
    max_cuda_version: u8,
    pub params: Box<[Param]>,
    pub sig: Signature,
    pub injections: HookInjections,
}

impl HookFn {
    pub fn parse(args: TokenStream, input: TokenStream) -> Result<Self> {
        Self::new(HookAttrs::from_macro(args)?, syn::parse2(input)?)
    }

    fn new(attrs: HookAttrs, item: HookFnItem) -> Result<Self> {
        let HookFnItem { mut sig, injections } = item;

        let mut params = Vec::with_capacity(sig.inputs.len());
        for arg in mem::take(&mut sig.inputs) {
            params.push(Param::parse(arg)?);
        }

        if let Some(true) = attrs.is_async_api {
            check_async_api(&params, &sig.output, &injections)?;
        }

        if params
            .iter()
            .filter(|param| {
                matches!(
                    param.kind,
                    ParamKind::InputHandle(InputHandleOp::Modify | InputHandleOp::Destroy)
                        | ParamKind::OutputHandle
                )
            })
            .count()
            > 1
        {
            return Err(Error::new_spanned(
                sig.ident,
                "cannot create, modify or destroy multiple handles",
            ));
        }

        if is_create_handle(&params) {
            if params.iter().filter(|p| p.is_host_output()).count() > 1 {
                return Err(Error::new_spanned(
                    sig.ident,
                    "cannot create handle with other host outputs",
                ));
            }
            if let Some(stmt) = injections.stmt_after_async_api_return() {
                return Err(Error::new_spanned(
                    stmt,
                    "cannot create handle with `after` injections",
                ));
            }
        }

        Ok(Self {
            proc_id: attrs.proc_id,
            is_async_api: attrs.is_async_api,
            parent: attrs.parent,
            min_cuda_version: attrs.min_cuda_version,
            max_cuda_version: attrs.max_cuda_version,
            params: params.into_boxed_slice(),
            sig,
            injections,
        })
    }

    pub fn func(&self) -> &Ident {
        &self.sig.ident
    }

    pub fn return_type(&self) -> Cow<'_, Type> {
        match &self.sig.output {
            ReturnType::Default => Cow::Owned(parse_quote! { () }),
            ReturnType::Type(_, ty) => Cow::Borrowed(ty.as_ref()),
        }
    }

    pub fn is_async_api(&self) -> bool {
        self.is_async_api.unwrap_or(false)
    }

    pub fn is_create_handle(&self) -> bool {
        is_create_handle(&self.params)
    }

    pub fn is_modify_handle(&self) -> bool {
        self.params
            .iter()
            .any(|param| matches!(param.kind, ParamKind::InputHandle(InputHandleOp::Modify)))
    }

    pub fn into_plain_fn(self) -> TokenStream {
        let mut sig = self.sig;

        if self.is_async_api.is_none()
            && check_async_api(&self.params, &sig.output, &self.injections).is_ok()
        {
            sig.ident.span().unwrap().note("this function can be `async_api`").emit();
        }

        for param in self.params {
            sig.inputs.push(param.into_plain_arg());
        }

        sig.ident =
            format_ident!("{}_{}_{}", sig.ident, self.min_cuda_version, self.max_cuda_version);

        quote! {
            #sig {
                unimplemented!()
            }
        }
    }
}

fn is_create_handle(params: &[Param]) -> bool {
    params.iter().any(|param| matches!(param.kind, ParamKind::OutputHandle))
}

fn check_async_api(
    params: &[Param],
    output: &ReturnType,
    injections: &HookInjections,
) -> Result<()> {
    if let Some(stmt) = injections.stmt_after_async_api_return() {
        return Err(Error::new_spanned(stmt, "`async_api` cannot have `after` injections"));
    }
    if let Some(param) = params.iter().find(|param| param.is_host_output()) {
        return Err(Error::new_spanned(&param.name, "`async_api` cannot have host outputs"));
    }
    match output {
        ReturnType::Type(_, ty) if !is_async_return_type(ty) => {
            Err(Error::new_spanned(output, "unsupported `async_api` return type"))
        }
        _ => Ok(()),
    }
}

pub struct Param {
    pub name: Ident,
    pub name_str: String,
    exe_ptr: Option<Ident>,
    colon: Token![:],
    ty: Box<Type>,
    pub kind: ParamKind,
}

impl Param {
    fn parse(arg: FnArg) -> Result<Self> {
        let FnArg::Typed(arg) = arg else { panic!() };

        // Get param name
        let Pat::Ident(PatIdent { by_ref: None, mutability: None, ident, subpat: None, .. }) =
            *arg.pat
        else {
            panic!()
        };

        check_max_attributes(&arg.attrs, 1)?;
        let kind = ParamKind::parse(arg.attrs.into_iter().next(), arg.ty.as_ref())?;

        Ok(Self {
            name_str: ident.to_string(),
            name: ident,
            exe_ptr: None,
            colon: arg.colon_token,
            ty: arg.ty,
            kind,
        })
    }

    fn into_plain_arg(self) -> FnArg {
        FnArg::Typed(PatType {
            attrs: Vec::new(),
            colon_token: self.colon,
            pat: Box::new(Pat::Ident(PatIdent {
                attrs: Vec::new(),
                by_ref: None,
                mutability: None,
                ident: self.name,
                subpat: None,
            })),
            ty: self.ty,
        })
    }

    pub fn raw_type(&self) -> &Type {
        &self.ty
    }

    pub fn deref_type(&self) -> &Type {
        let Type::Ptr(ptr) = self.raw_type() else { panic!() };
        ptr.elem.as_ref()
    }

    pub fn is_host_output(&self) -> bool {
        self.kind.is_host_output()
    }

    pub fn is_hacked_type(&self) -> bool {
        is_hacked_type(&self.ty)
    }

    pub fn exe_ptr(&self) -> &Ident {
        self.exe_ptr.as_ref().unwrap()
    }

    pub fn setup_exe_ptr(&mut self) {
        self.exe_ptr = Some(format_ident!("{}__ptr", self.name));
    }
}

#[derive(Debug)]
pub enum ParamKind {
    InputValue,
    InputHandle(InputHandleOp),
    InputSinglePtr,
    InputArrayPtr { is_void_ptr: bool, len: Box<Expr> },
    InputCStr,
    OutputHandle,
    OutputSinglePtr,
    OutputArrayPtr { is_void_ptr: bool, len: Box<Expr>, cap: Option<Box<Expr>> },
    DeviceInputPtr,
    DeviceOutputPtr, // TODO: use when we support CoW and recopy
    Skip,
    Const(Box<Expr>),
}

#[derive(Debug)]
pub enum InputHandleOp {
    Use,
    Modify,
    Destroy,
}

impl InputHandleOp {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "use" => Some(Self::Use),
            "modify" => Some(Self::Modify),
            "destroy" => Some(Self::Destroy),
            _ => None,
        }
    }
}

impl ParamKind {
    fn parse(attr: Option<Attribute>, ty: &Type) -> Result<Self> {
        let message = "expected #[skip] or #[value = ...]";

        let attr = match attr {
            Some(attr) => match attr.path().require_ident()?.to_string().as_str() {
                "skip" => match attr.meta {
                    Meta::Path(_) => return Ok(Self::Skip),
                    _ => return Err(Error::new_spanned(attr, message)),
                },
                "value" => match attr.meta {
                    Meta::NameValue(meta) => return Ok(Self::Const(Box::new(meta.value))),
                    _ => return Err(Error::new_spanned(attr, message)),
                },
                _ => Some(attr),
            },
            None => None,
        };

        if let Type::Ptr(ptr) = ty {
            return Self::parse_ptr(attr, ptr);
        }

        if is_handle_type(ty) {
            return if let Some(attr) = &attr
                && let Some(op) = InputHandleOp::parse(&parse_handle_op(attr)?)
            {
                Ok(Self::InputHandle(op))
            } else {
                Err(Error::new_spanned(
                    ty,
                    "requires #[handle = \"...\"] (\"use\", \"modify\" or \"destroy\")",
                ))
            };
        }

        match attr {
            None => Ok(Self::InputValue),
            Some(attr) => Err(Error::new_spanned(attr, message)),
        }
    }

    fn parse_ptr(attr: Option<Attribute>, ptr: &TypePtr) -> Result<Self> {
        if is_handle_type(&ptr.elem) {
            if ptr.const_token.is_some() {
                return Err(Error::new_spanned(ptr, "unexpected *const handle"));
            }
            return if let Some(attr) = &attr
                && parse_handle_op(attr)? == "create"
            {
                Ok(Self::OutputHandle)
            } else {
                Err(Error::new_spanned(ptr, "requires #[handle = \"create\"]"))
            };
        }

        if let Some(attr) = attr {
            return parse_ptr_attr(&attr, ptr);
        }

        if is_const_cstr(ptr) {
            return Ok(Self::InputCStr);
        }

        if ptr.const_token.is_some() || is_void_ptr(ptr) {
            return Err(Error::new_spanned(ptr, "requires #[device] or #[host(...)]"));
        }

        Ok(Self::OutputSinglePtr)
    }

    fn is_host_output(&self) -> bool {
        match self {
            Self::OutputHandle | Self::OutputSinglePtr | Self::OutputArrayPtr { .. } => true,
            Self::InputValue
            | Self::InputHandle(_)
            | Self::InputSinglePtr
            | Self::InputArrayPtr { .. }
            | Self::InputCStr
            | Self::DeviceInputPtr
            | Self::DeviceOutputPtr
            | Self::Skip
            | Self::Const(_) => false,
        }
    }
}

fn parse_ptr_attr(attr: &Attribute, ptr: &TypePtr) -> Result<ParamKind> {
    enum Mode {
        None,
        Input,
        Output,
    }
    enum Ptr {
        Const,
        Mut,
    }
    let location = attr.path().require_ident()?.to_string();
    if location == "host" {
        let is_void_ptr = is_void_ptr(ptr);
        let mut mode = Mode::None;
        let mut len = None;
        let mut cap = None;
        if matches!(attr.meta, Meta::Path(_)) {
            // No nested meta
        } else {
            attr.parse_nested_meta(|meta| {
                match meta.path.require_ident()?.to_string().as_str() {
                    "input" => mode = Mode::Input,
                    "output" => mode = Mode::Output,
                    "len" => len = Some(meta.value()?.parse()?),
                    "cap" => cap = Some(meta.value()?.parse()?),
                    _ => return Err(meta.error("unsupported property")),
                }
                Ok(())
            })?;
        }

        if is_void_ptr && len.is_none() {
            return Err(Error::new_spanned(attr, "len property is required for void pointer"));
        }

        let ptr_type = match ptr.const_token {
            Some(_) => Ptr::Const,
            None => Ptr::Mut,
        };

        match (ptr_type, mode) {
            (Ptr::Const, Mode::None) | (Ptr::Mut, Mode::Input) => match (len, cap) {
                (None, None) => Ok(ParamKind::InputSinglePtr),
                (Some(len), None) => Ok(ParamKind::InputArrayPtr { is_void_ptr, len }),
                (_, Some(_)) => Err(Error::new_spanned(attr, "input array cannot have cap")),
            },
            (Ptr::Const, Mode::Input | Mode::Output) => {
                Err(Error::new_spanned(attr, "input/output is not allowed on const pointer"))
            }
            (Ptr::Mut, Mode::Output) => match (len, cap) {
                (None, None) => Ok(ParamKind::OutputSinglePtr),
                (Some(len), cap) => Ok(ParamKind::OutputArrayPtr { is_void_ptr, len, cap }),
                (None, Some(_)) => Err(Error::new_spanned(attr, "only cap is specified")),
            },
            (Ptr::Mut, Mode::None) => {
                Err(Error::new_spanned(attr, "input/output is required for mutable pointer"))
            }
        }
    } else if location == "device" && matches!(attr.meta, Meta::Path(_)) {
        Ok(ParamKind::DeviceInputPtr) // TODO: DeviceOutputPtr (based on *mut/const, maybe allow override?)
    } else {
        Err(Error::new_spanned(attr, "expected #[device] or #[host(...)]"))
    }
}

fn parse_handle_op(attr: &Attribute) -> Result<String> {
    if attr.path().require_ident()? == "handle"
        && let Expr::Lit(expr) = &attr.meta.require_name_value()?.value
        && let Lit::Str(op) = &expr.lit
    {
        Ok(op.value())
    } else {
        Err(Error::new_spanned(attr, "expected #[handle = \"...\"]"))
    }
}
