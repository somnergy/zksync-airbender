use field::PrimeField;
use std::mem::MaybeUninit;

pub mod evaluation_kernel;
pub mod storage;

// for notations: if we have a poly p(x0, x1, x2, x3, ...) then x0 is the most signinicant bit in the
// indexing of the array

pub trait LazyPolyFolder<'a, F: PrimeField> {
    type InputBuffer: Sized;
    type Intermediate: Sized;
    type NextStepLazyBuffer: Sized;

    fn construct(
        input: Self::InputBuffer,
        intermediate: Self::Intermediate,
        next_step: Self::NextStepLazyBuffer,
        repeated: bool,
    ) -> Self;

    fn get_intermediate_only(&mut self, index: usize) -> F {
        self.get_triple(index).2
    }
    fn get_triple(&mut self, index: usize) -> (F, F, F);
    fn size(&self) -> usize;
}

pub struct FirstStepInExtensionLazyFolder<'a, F: PrimeField> {
    input_evaluations: *const F,
    next_step_buffer: *mut F,
    half_size_offset: usize,
    next_buffer_is_filled: bool,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a, F: PrimeField> LazyPolyFolder<'a, F> for FirstStepInExtensionLazyFolder<'a, F> {
    type InputBuffer = &'a [F];
    type Intermediate = ();
    type NextStepLazyBuffer = &'a mut [MaybeUninit<F>];

    fn construct(
        input: Self::InputBuffer,
        _intermediate: Self::Intermediate,
        next_step: Self::NextStepLazyBuffer,
        repeated: bool,
    ) -> Self {
        let size = input.len();
        assert!(size.is_power_of_two());
        let half_size_offset = size / 2;

        Self {
            input_evaluations: input.as_ptr(),
            next_step_buffer: next_step.as_mut_ptr().cast(),
            half_size_offset,
            next_buffer_is_filled: repeated,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn get_triple(&mut self, index: usize) -> (F, F, F) {
        debug_assert!(index < self.half_size_offset);
        unsafe {
            let f0 = self.input_evaluations.add(index).read();
            let f1 = self
                .input_evaluations
                .add(self.half_size_offset + index)
                .read();
            let mut lazy_eval;
            if self.next_buffer_is_filled {
                lazy_eval = self.next_step_buffer.add(index).cast::<F>().read();
            } else {
                lazy_eval = f1;
                lazy_eval.sub_assign(&f0);
                self.next_step_buffer.add(index).write(lazy_eval);
            }

            (f0, f1, lazy_eval)
        }
    }

    fn size(&self) -> usize {
        self.half_size_offset * 2
    }
}

pub struct NextStepsInExtensionLazyFolder<'a, F: PrimeField> {
    previous_step_at_0_evaluations: *const F,
    input_evaluations: *mut F,
    next_step_buffer: *mut F,
    half_size_offset: usize,
    next_buffer_is_filled: bool,
    folding_challenge: F,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a, F: PrimeField> LazyPolyFolder<'a, F> for NextStepsInExtensionLazyFolder<'a, F> {
    type InputBuffer = (&'a mut [F], &'a [F]);
    type Intermediate = F;
    type NextStepLazyBuffer = &'a mut [MaybeUninit<F>];

    fn construct(
        input: Self::InputBuffer,
        intermediate: Self::Intermediate,
        next_step: Self::NextStepLazyBuffer,
        repeated: bool,
    ) -> Self {
        let (this_step_lazy_buffer, previous_step_evals_at_0) = input;
        let size = this_step_lazy_buffer.len();
        assert!(size.is_power_of_two());
        assert_eq!(size, previous_step_evals_at_0.len());
        let half_size_offset = size / 2;

        Self {
            previous_step_at_0_evaluations: previous_step_evals_at_0.as_ptr(),
            input_evaluations: this_step_lazy_buffer.as_mut_ptr(),
            next_step_buffer: next_step.as_mut_ptr().cast(),
            half_size_offset,
            next_buffer_is_filled: repeated,
            folding_challenge: intermediate,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn get_triple(&mut self, index: usize) -> (F, F, F) {
        debug_assert!(index < self.half_size_offset);
        unsafe {
            if self.next_buffer_is_filled {
                let f0 = self.input_evaluations.add(index).read();
                let f1 = self
                    .input_evaluations
                    .add(self.half_size_offset + index)
                    .read();
                let lazy_eval = self.next_step_buffer.add(index).cast::<F>().read();
                (f0, f1, lazy_eval)
            } else {
                // NOTE: in general we only need to materialize f0 for the next folding step, but
                // as we do not yet differentiate if we will access this poly once or not, then we will also write down f(1) too for future
                // repeated reads

                // finish folding from the previous step. We expect f(1) - f(0) in the buffer, and need
                // to finish computing f(0) + challenge * (f(1) - f(0))
                let mut f0 = self.input_evaluations.add(index).cast::<F>().read();
                f0.mul_assign(&self.folding_challenge);
                f0.add_assign(
                    self.previous_step_at_0_evaluations
                        .add(index)
                        .as_ref_unchecked(),
                );
                self.input_evaluations.add(index).write(f0);

                let mut f1 = self
                    .input_evaluations
                    .add(self.half_size_offset + index)
                    .cast::<F>()
                    .read();
                f1.mul_assign(&self.folding_challenge);
                f1.add_assign(
                    self.previous_step_at_0_evaluations
                        .add(self.half_size_offset + index)
                        .as_ref_unchecked(),
                );
                self.input_evaluations
                    .add(self.half_size_offset + index)
                    .write(f1);

                let mut lazy_eval = f1;
                lazy_eval.sub_assign(&f0);
                self.next_step_buffer.add(index).write(lazy_eval);

                (f0, f1, lazy_eval)
            }
        }
    }

    fn size(&self) -> usize {
        self.half_size_offset * 2
    }
}

// same ones would be needed for the case when we fold base layer poly
