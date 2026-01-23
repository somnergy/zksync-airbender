use crate::field::{BaseField, Ext2Field};
use era_cudart::cuda_kernel;
use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::execution::{CudaLaunchConfig, KernelFunction};
use era_cudart::result::CudaResult;
use era_cudart::stream::CudaStream;
use era_cudart_sys::CudaDeviceAttr::MultiProcessorCount;
use std::ptr::null_mut;

type BF = BaseField;
type E2 = Ext2Field;

cuda_kernel!(BenchBase, bench_bf, values: *const BF, count: u32);

bench_bf!(ab_bf_add_bench_kernel);
bench_bf!(ab_bf_mul_bench_kernel);

pub fn bf_add_bench(stream: &CudaStream) -> CudaResult<()> {
    let device_id = get_device()?;
    let mpc = device_get_attribute(MultiProcessorCount, device_id)? as u32;
    let config = CudaLaunchConfig::basic(mpc, 1024, stream);
    let args = BenchBaseArguments::new(null_mut(), 0);
    BenchBaseFunction(ab_bf_add_bench_kernel).launch(&config, &args)
}

pub fn bf_mul_bench(stream: &CudaStream) -> CudaResult<()> {
    let device_id = get_device()?;
    let mpc = device_get_attribute(MultiProcessorCount, device_id)? as u32;
    let config = CudaLaunchConfig::basic(mpc, 1024, stream);
    let args = BenchBaseArguments::new(null_mut(), 0);
    BenchBaseFunction(ab_bf_mul_bench_kernel).launch(&config, &args)
}

cuda_kernel!(BenchExt2, bench_e2, values: *const E2, count: u32);

bench_e2!(ab_e2_sqr_bench_kernel);

pub fn e2_sqr_bench(stream: &CudaStream) -> CudaResult<()> {
    let device_id = get_device()?;
    let mpc = device_get_attribute(MultiProcessorCount, device_id)? as u32;
    let config = CudaLaunchConfig::basic(mpc, 1024, stream);
    let args = BenchExt2Arguments::new(null_mut(), 0);
    BenchExt2Function(ab_e2_sqr_bench_kernel).launch(&config, &args)
}
