use crate::{baby_bear::base::BabyBearField, Mersenne31Field};

impl quote::ToTokens for Mersenne31Field {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use quote::quote;
        let inner = self.to_reduced_u32();
        let t = quote! { Mersenne31Field(#inner) };
        tokens.extend(t);
    }
}

impl quote::ToTokens for BabyBearField {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use quote::quote;
        let inner = self.raw_u32_value();
        let t = quote! { BabyBearField(#inner) };
        tokens.extend(t);
    }
}
