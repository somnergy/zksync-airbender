use std::mem::size_of;
use std::os::raw::c_void;

use era_cudart::memory::{memory_copy, DeviceAllocation};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::slice::DeviceSlice;
use era_cudart_sys::{cudaMemcpyToSymbol, cuda_struct_and_stub, CudaMemoryCopyKind};
use fft::bitreverse_enumeration_inplace;
use fft::field_utils::{distribute_powers_serial, domain_generator_for_size};
use field::Field;

use crate::field::BaseField;

type BF = BaseField;

pub const TWO_ADICITY: u32 = 27;
pub const NTT_COARSE_TWO_ADIC_LOG_COUNT: u32 = 13;
// "- 1" accounts for NTT twiddle arrays only covering half the range
pub const NTT_FINE_TWO_ADIC_LOG_COUNT: u32 = TWO_ADICITY - NTT_COARSE_TWO_ADIC_LOG_COUNT - 1;
pub const CMEM_TWO_ADICITY: u32 = 19;
pub const CMEM_COARSE_LOG_COUNT: u32 = 10;
pub const CMEM_FINE_LOG_COUNT: u32 = CMEM_TWO_ADICITY - CMEM_COARSE_LOG_COUNT - 1;
pub const LENGTH_CMEM_COARSE: u32 = 1 << CMEM_COARSE_LOG_COUNT;

#[repr(C)]
struct PowersLayerData {
    values: *const BF,
    mask: u32,
    log_count: u32,
}

impl PowersLayerData {
    fn new(values: *const BF, log_count: u32) -> Self {
        let mask = (1 << log_count) - 1;
        Self {
            values,
            mask,
            log_count,
        }
    }
}

#[cfg(no_cuda)]
unsafe impl Sync for PowersLayerData {}

#[repr(C)]
struct PowersData2Layer {
    fine: PowersLayerData,
    coarse: PowersLayerData,
}

impl PowersData2Layer {
    fn new(
        fine_values: *const BF,
        fine_log_count: u32,
        coarse_values: *const BF,
        coarse_log_count: u32,
    ) -> Self {
        let fine = PowersLayerData::new(fine_values, fine_log_count);
        let coarse = PowersLayerData::new(coarse_values, coarse_log_count);
        Self { fine, coarse }
    }
}

#[cfg(no_cuda)]
unsafe impl Sync for PowersData2Layer {}

// #[repr(C)]
// struct PowersData3Layer {
//     fine: PowersLayerData,
//     coarser: PowersLayerData,
//     coarsest: PowersLayerData,
// }
//
// impl PowersData3Layer {
//     fn new(
//         fine_values: *const BF,
//         fine_log_count: u32,
//         coarser_values: *const BF,
//         coarser_log_count: u32,
//         coarsest_values: *const BF,
//         coarsest_log_count: u32,
//     ) -> Self {
//         let fine = PowersLayerData::new(fine_values, fine_log_count);
//         let coarser = PowersLayerData::new(coarser_values, coarser_log_count);
//         let coarsest = PowersLayerData::new(coarsest_values, coarsest_log_count);
//         Self {
//             fine,
//             coarser,
//             coarsest,
//         }
//     }
// }
//
// #[cfg(no_cuda)]
// unsafe impl Sync for PowersData3Layer {}

// cuda_struct_and_stub! { static ab_powers_data_w: PowersData3Layer; }
cuda_struct_and_stub! { static ab_powers_data_w_bitrev_for_ntt: PowersData2Layer; }
cuda_struct_and_stub! { static ab_powers_data_w_inv_bitrev_for_ntt: PowersData2Layer; }
cuda_struct_and_stub! { static ab_inv_sizes: [BF; TWO_ADICITY as usize]; }
cuda_struct_and_stub! { static ab_fwd_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_inv_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_fwd_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize]; }
cuda_struct_and_stub! { static ab_inv_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize]; }

unsafe fn copy_to_symbol<T>(symbol: &T, src: &T) -> CudaResult<()> {
    cudaMemcpyToSymbol(
        symbol as *const T as *const c_void,
        src as *const T as *const c_void,
        size_of::<T>(),
        0,
        CudaMemoryCopyKind::HostToDevice,
    )
    .wrap()
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy_to_symbols(
    // powers_of_w_coarsest_log_count: u32,
    // powers_of_w_fine: *const BF,
    // powers_of_w_coarser: *const BF,
    // powers_of_w_coarsest: *const BF,
    powers_of_w_fine_bitrev_for_ntt: *const BF,
    powers_of_w_coarse_bitrev_for_ntt: *const BF,
    powers_of_w_inv_fine_bitrev_for_ntt: *const BF,
    powers_of_w_inv_coarse_bitrev_for_ntt: *const BF,
    inv_sizes_host: [BF; TWO_ADICITY as usize],
    fwd_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize],
    inv_cmem_twiddles_coarse: [BF; 1 << CMEM_COARSE_LOG_COUNT as usize],
    fwd_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize],
    inv_cmem_twiddles_fine: [BF; 1 << CMEM_FINE_LOG_COUNT as usize],
) -> CudaResult<()> {
    // let coarsest_log_count = powers_of_w_coarsest_log_count;
    // let coarser_log_count = OMEGA_LOG_ORDER - coarsest_log_count;
    // copy_to_symbol(
    //     &ab_powers_data_w,
    //     &PowersData3Layer::new(
    //         powers_of_w_fine,
    //         FINEST_LOG_COUNT,
    //         powers_of_w_coarser,
    //         coarser_log_count,
    //         powers_of_w_coarsest,
    //         coarsest_log_count,
    //     ),
    // )?;
    let fine_log_count = NTT_FINE_TWO_ADIC_LOG_COUNT;
    let coarse_log_count = NTT_COARSE_TWO_ADIC_LOG_COUNT;
    copy_to_symbol(
        &ab_powers_data_w_bitrev_for_ntt,
        &PowersData2Layer::new(
            powers_of_w_fine_bitrev_for_ntt,
            fine_log_count,
            powers_of_w_coarse_bitrev_for_ntt,
            coarse_log_count,
        ),
    )?;
    copy_to_symbol(
        &ab_powers_data_w_inv_bitrev_for_ntt,
        &PowersData2Layer::new(
            powers_of_w_inv_fine_bitrev_for_ntt,
            fine_log_count,
            powers_of_w_inv_coarse_bitrev_for_ntt,
            coarse_log_count,
        ),
    )?;
    copy_to_symbol(&ab_inv_sizes, &inv_sizes_host)?;
    copy_to_symbol(&ab_fwd_cmem_twiddles_coarse, &fwd_cmem_twiddles_coarse)?;
    copy_to_symbol(&ab_inv_cmem_twiddles_coarse, &inv_cmem_twiddles_coarse)?;
    copy_to_symbol(&ab_fwd_cmem_twiddles_fine, &fwd_cmem_twiddles_fine)?;
    copy_to_symbol(&ab_inv_cmem_twiddles_fine, &inv_cmem_twiddles_fine)?;
    Ok(())
}

fn generate_powers_dev<F: Field>(
    base: F,
    powers_dev: &mut DeviceSlice<F>,
    bit_reverse: bool,
) -> CudaResult<()> {
    let mut powers_host = vec![F::ONE; powers_dev.len()];
    distribute_powers_serial::<F, F>(&mut powers_host, F::ONE, base);
    if bit_reverse {
        bitreverse_enumeration_inplace(&mut powers_host);
    }
    memory_copy(powers_dev, &powers_host)
}

pub struct DeviceContext {
    // pub powers_of_w_fine: DeviceAllocation<BF>,
    // pub powers_of_w_coarser: DeviceAllocation<BF>,
    // pub powers_of_w_coarsest: DeviceAllocation<BF>,
    pub powers_of_w_fine_bitrev_for_ntt: DeviceAllocation<BF>,
    pub powers_of_w_coarse_bitrev_for_ntt: DeviceAllocation<BF>,
    pub powers_of_w_inv_fine_bitrev_for_ntt: DeviceAllocation<BF>,
    pub powers_of_w_inv_coarse_bitrev_for_ntt: DeviceAllocation<BF>,
}

impl DeviceContext {
    pub fn create(powers_of_w_coarse_log_count: u32) -> CudaResult<Self> {
        // assert!(powers_of_w_coarsest_log_count <= TWO_ADICITY);
        // let length_fine = 1usize << FINEST_LOG_COUNT;
        // let length_coarser = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarsest_log_count);
        // let length_coarsest = 1usize << powers_of_w_coarsest_log_count;
        // let mut powers_of_w_fine = DeviceAllocation::<BF>::alloc(length_fine)?;
        // let mut powers_of_w_coarser = DeviceAllocation::<BF>::alloc(length_coarser)?;
        // let mut powers_of_w_coarsest = DeviceAllocation::<BF>::alloc(length_coarsest)?;
        // generate_powers_dev(
        //     domain_generator_for_size::<BF>(1u64 << CIRCLE_GROUP_LOG_ORDER),
        //     &mut powers_of_w_fine,
        //     false,
        // )?;
        // generate_powers_dev(
        //     domain_generator_for_size::<BF>(1u64 << OMEGA_LOG_ORDER),
        //     &mut powers_of_w_coarser,
        //     false,
        // )?;
        // generate_powers_dev(
        //     domain_generator_for_size::<BF>(length_coarsest as u64),
        //     &mut powers_of_w_coarsest,
        //     false,
        // )?;
        let length_fine = 1usize << NTT_FINE_TWO_ADIC_LOG_COUNT;
        let length_coarse = 1usize << NTT_COARSE_TWO_ADIC_LOG_COUNT;
        let mut powers_of_w_fine_bitrev_for_ntt = DeviceAllocation::<BF>::alloc(length_fine)?;
        let mut powers_of_w_coarse_bitrev_for_ntt = DeviceAllocation::<BF>::alloc(length_coarse)?;
        let mut powers_of_w_inv_fine_bitrev_for_ntt = DeviceAllocation::<BF>::alloc(length_fine)?;
        let mut powers_of_w_inv_coarse_bitrev_for_ntt =
            DeviceAllocation::<BF>::alloc(length_coarse)?;
        generate_powers_dev(
            domain_generator_for_size::<BF>(1u64 << TWO_ADICITY),
            &mut powers_of_w_fine_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<BF>((length_coarse * 2) as u64),
            &mut powers_of_w_coarse_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<BF>(1u64 << TWO_ADICITY)
                .inverse()
                .expect("must exist"),
            &mut powers_of_w_inv_fine_bitrev_for_ntt,
            true,
        )?;
        generate_powers_dev(
            domain_generator_for_size::<BF>((length_coarse * 2) as u64)
                .inverse()
                .expect("must exist"),
            &mut powers_of_w_inv_coarse_bitrev_for_ntt,
            true,
        )?;
        let two_inv = BF::new(2).inverse().expect("must exist");
        let mut inv_sizes_host = [BF::ONE; TWO_ADICITY as usize];
        distribute_powers_serial(&mut inv_sizes_host, BF::ONE, two_inv);
        // trust me
        let generator_fwd_cmem_coarse =
            domain_generator_for_size::<BF>((LENGTH_CMEM_COARSE * 2) as u64);
        let mut fwd_cmem_twiddles_coarse = [BF::ONE; 1 << CMEM_COARSE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut fwd_cmem_twiddles_coarse,
            BF::ONE,
            generator_fwd_cmem_coarse,
        );
        bitreverse_enumeration_inplace(&mut fwd_cmem_twiddles_coarse);
        let generator_inv_cmem_coarse = generator_fwd_cmem_coarse.inverse().expect("must exist");
        let mut inv_cmem_twiddles_coarse = [BF::ONE; 1 << CMEM_COARSE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut inv_cmem_twiddles_coarse,
            BF::ONE,
            generator_inv_cmem_coarse,
        );
        bitreverse_enumeration_inplace(&mut inv_cmem_twiddles_coarse);
        let generator_fwd_cmem_fine = domain_generator_for_size::<BF>(1u64 << CMEM_TWO_ADICITY);
        let mut fwd_cmem_twiddles_fine = [BF::ONE; 1 << CMEM_FINE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut fwd_cmem_twiddles_fine,
            BF::ONE,
            generator_fwd_cmem_fine,
        );
        bitreverse_enumeration_inplace(&mut fwd_cmem_twiddles_fine);
        let generator_inv_cmem_fine = generator_fwd_cmem_fine.inverse().expect("must exist");
        let mut inv_cmem_twiddles_fine = [BF::ONE; 1 << CMEM_FINE_LOG_COUNT as usize];
        distribute_powers_serial(
            &mut inv_cmem_twiddles_fine,
            BF::ONE,
            generator_inv_cmem_fine,
        );
        bitreverse_enumeration_inplace(&mut inv_cmem_twiddles_fine);
        unsafe {
            copy_to_symbols(
                // powers_of_w_coarsest_log_count,
                // powers_of_w_fine.as_ptr(),
                // powers_of_w_coarser.as_ptr(),
                // powers_of_w_coarsest.as_ptr(),
                powers_of_w_fine_bitrev_for_ntt.as_ptr(),
                powers_of_w_coarse_bitrev_for_ntt.as_ptr(),
                powers_of_w_inv_fine_bitrev_for_ntt.as_ptr(),
                powers_of_w_inv_coarse_bitrev_for_ntt.as_ptr(),
                inv_sizes_host,
                fwd_cmem_twiddles_coarse,
                inv_cmem_twiddles_coarse,
                fwd_cmem_twiddles_fine,
                inv_cmem_twiddles_fine,
            )?;
        }
        Ok(Self {
            // powers_of_w_fine,
            // powers_of_w_coarser,
            // powers_of_w_coarsest,
            powers_of_w_fine_bitrev_for_ntt,
            powers_of_w_coarse_bitrev_for_ntt,
            powers_of_w_inv_fine_bitrev_for_ntt,
            powers_of_w_inv_coarse_bitrev_for_ntt,
        })
    }

    pub fn destroy(self) -> CudaResult<()> {
        // self.powers_of_w_fine.free()?;
        // self.powers_of_w_coarser.free()?;
        // self.powers_of_w_coarsest.free()?;
        self.powers_of_w_fine_bitrev_for_ntt.free()?;
        self.powers_of_w_coarse_bitrev_for_ntt.free()?;
        self.powers_of_w_inv_fine_bitrev_for_ntt.free()?;
        self.powers_of_w_inv_coarse_bitrev_for_ntt.free()?;
        Ok(())
    }
}
