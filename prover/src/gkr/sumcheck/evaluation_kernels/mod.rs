use field::{FieldExtension, PrimeField};

pub mod batch_constraint_eval_example;
pub mod trivial_product_in_extension;

pub trait EvaluationRepresentation<F: PrimeField, E: FieldExtension<F> + PrimeField>:
    'static + Clone + Copy + core::fmt::Debug + Send + Sync
{
    type CollapseContext: 'static + Clone + Copy + core::fmt::Debug + Send + Sync;
    type CollapsedForm: PrimeField;
    fn collapse(self, ctx: &Self::CollapseContext) -> Self::CollapsedForm;

    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E;

    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self);
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E> for () {
    type CollapseContext = ();
    type CollapsedForm = F;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        F::ZERO
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, _ctx: &Self::CollapseContext, _challenge: &E) -> E {
        E::ZERO
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, _other: &Self) {}
}

#[derive(Clone, Copy, Debug)]
pub struct BaseFieldRepresentation<F: PrimeField>(pub(crate) F);

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for BaseFieldRepresentation<F>
{
    type CollapseContext = ();
    type CollapsedForm = F;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        self.0
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, _ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = *challenge;
        result.mul_assign_by_base(&self.0);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.0.mul_assign(&other.0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BaseFieldFoldedOnceRepresentation<F: PrimeField> {
    pub(crate) c0: F,
    pub(crate) c1: F,
    pub(crate) computed_r2_coeff: F,
}

impl<F: PrimeField> BaseFieldFoldedOnceRepresentation<F> {
    #[inline(always)]
    pub fn new(c0: F, c1: F) -> Self {
        Self {
            c0,
            c1,
            computed_r2_coeff: F::ZERO,
        }
    }
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for BaseFieldFoldedOnceRepresentation<F>
{
    type CollapseContext = (E, E);
    type CollapsedForm = E;
    #[inline(always)]
    fn collapse(self, ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        let (mut r, r2) = *ctx;
        let mut result = r2;
        result.mul_assign_by_base(&self.computed_r2_coeff);
        r.mul_assign_by_base(&self.c1);
        result.add_assign(&r);
        result.add_assign_base(&self.c0);

        result
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = self.collapse(ctx);
        result.mul_assign(challenge);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.c0.add_assign(&other.c0);
        self.c1.add_assign(&other.c1);
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            self.computed_r2_coeff.add_assign(&other.computed_r2_coeff);
        }
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.c0.sub_assign(&other.c0);
        self.c1.sub_assign(&other.c1);
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            self.computed_r2_coeff.sub_assign(&other.computed_r2_coeff);
        }
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        if ASSUME_NO_PRODUCTS_BEFORE == false {
            panic!();
        }
        self.computed_r2_coeff = self.c1;
        self.computed_r2_coeff.mul_assign(&other.c0);
        self.c1.mul_assign(&other.c0);
        let mut tt = self.c0;
        tt.mul_assign(&other.c1);
        self.c1.add_assign(&tt);

        self.c0.mul_assign(&other.c0);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExtensionFieldRepresentation<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub(crate) value: E,
    pub(crate) _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> ExtensionFieldRepresentation<F, E> {
    #[inline(always)]
    pub fn new(value: E) -> Self {
        Self {
            value,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> EvaluationRepresentation<F, E>
    for ExtensionFieldRepresentation<F, E>
{
    type CollapseContext = ();
    type CollapsedForm = E;
    #[inline(always)]
    fn collapse(self, _ctx: &Self::CollapseContext) -> Self::CollapsedForm {
        self.value
    }
    #[inline(always)]
    fn collapse_for_batch_eval(self, ctx: &Self::CollapseContext, challenge: &E) -> E {
        let mut result = self.collapse(ctx);
        result.mul_assign(challenge);
        result
    }
    #[inline(always)]
    fn repr_add_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.add_assign(&other.value);
    }
    #[inline(always)]
    fn repr_sub_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.sub_assign(&other.value);
    }
    #[inline(always)]
    fn repr_mul_assign<const ASSUME_NO_PRODUCTS_BEFORE: bool>(&mut self, other: &Self) {
        self.value.mul_assign(&other.value);
    }
}

pub trait EvaluationFormStorage<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    R: EvaluationRepresentation<F, E>,
>
{
    fn dummy() -> Self;
    fn get_collapse_context(&self) -> &R::CollapseContext;
    fn get_f0_and_f1_minus_f0(&self, index: usize) -> [R; 2];
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField, R: EvaluationRepresentation<F, E>>
    EvaluationFormStorage<F, E, R> for ()
{
    fn dummy() -> Self {
        ()
    }
    #[inline(always)]
    fn get_collapse_context(&self) -> &R::CollapseContext {
        unreachable!()
    }
    #[inline(always)]
    fn get_f0_and_f1_minus_f0(&self, _index: usize) -> [R; 2] {
        unreachable!()
    }
}

pub struct SumcheckAccumulatorDst<F: PrimeField, E: FieldExtension<F> + PrimeField> {
    pub(crate) dest: *mut [E; 2],
    pub(crate) _marker: core::marker::PhantomData<F>,
}

unsafe impl<F: PrimeField, E: FieldExtension<F> + PrimeField> Send
    for SumcheckAccumulatorDst<F, E>
{
}

impl<F: PrimeField, E: FieldExtension<F> + PrimeField> SumcheckAccumulatorDst<F, E> {
    #[inline(always)]
    pub(crate) fn get_dst(&self, index: usize) -> &mut [E; 2] {
        unsafe { self.dest.add(index).as_mut_unchecked() }
    }
}

pub trait BatchSumcheckEvaluationKernel<
    F: PrimeField,
    E: FieldExtension<F> + PrimeField,
    R0: EvaluationRepresentation<F, E>, // inputs in the base field at the first sumcheck step
    R1: EvaluationRepresentation<F, E>, // inputs already in the extension field at the first sumcheck step
>
{
    fn evaluate<
        S0: EvaluationFormStorage<F, E, R0>,
        S1: EvaluationFormStorage<F, E, R1>,
        const FIRST_ROUND: bool,
    >(
        &self,
        index: usize,
        r0_sources: &[S0],
        r1_sources: &[S1],
        batch_challenge: &E,
    ) -> [E; 2];
}
