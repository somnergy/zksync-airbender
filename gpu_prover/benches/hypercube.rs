#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, Throughput};
use era_criterion_cuda::CudaMeasurement;
use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::memory::{memory_copy_async, DeviceAllocation};
use era_cudart::result::CudaResult;
use era_cudart::stream::CudaStream;
use era_cudart_sys::CudaDeviceAttr;
use field::Field;
use gpu_prover::field::BF;
use gpu_prover::{
    hypercube_evals_into_coeffs_bitrev_bf, hypercube_evals_into_coeffs_bitrev_bf_in_place,
};

const SUPPORTED_LOG_ROWS: [u32; 4] = [21, 22, 23, 24];
const L2_RING_TARGET_BYTES_MULTIPLIER: usize = 2;
const L2_RING_TARGET_MIN_BYTES: usize = 8 * 1024 * 1024;

fn stage1_ring_len(rows: usize) -> CudaResult<usize> {
    let row_bytes = rows * std::mem::size_of::<BF>();
    let device_id = get_device()?;
    let l2_bytes = device_get_attribute(CudaDeviceAttr::L2CacheSize, device_id)? as usize;
    let target_bytes = (l2_bytes * L2_RING_TARGET_BYTES_MULTIPLIER).max(L2_RING_TARGET_MIN_BYTES);
    let ring_len = ((target_bytes + row_bytes - 1) / row_bytes).max(2);
    Ok(ring_len)
}

struct HypercubeBitrevBenchCase {
    log_rows: u32,
    rows: usize,
    d_src: DeviceAllocation<BF>,
    d_dst: DeviceAllocation<BF>,
}

impl HypercubeBitrevBenchCase {
    fn new(log_rows: u32, stream: &CudaStream) -> CudaResult<Self> {
        let rows = 1usize << log_rows;

        let mut d_src = DeviceAllocation::alloc(rows)?;
        let d_dst = DeviceAllocation::alloc(rows)?;

        // Fill once to avoid benchmarking uninitialized memory reads.
        let h_src = vec![BF::ZERO; rows];
        memory_copy_async(&mut d_src, &h_src, stream)?;
        stream.synchronize()?;

        Ok(Self {
            log_rows,
            rows,
            d_src,
            d_dst,
        })
    }

    fn run_out_of_place(&mut self, stream: &CudaStream) -> CudaResult<()> {
        hypercube_evals_into_coeffs_bitrev_bf(&self.d_src, &mut self.d_dst, stream)
    }

    fn run_in_place(&mut self, stream: &CudaStream) -> CudaResult<()> {
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut self.d_src, stream)
    }

    fn bytes_per_transform(&self) -> u64 {
        // Approximate traffic: read + write per launch, with exactly 3 launches.
        (self.rows as u64) * (std::mem::size_of::<BF>() as u64) * 2 * 3
    }
}

fn benchmark_out_of_place(c: &mut Criterion<CudaMeasurement>) {
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("hypercube_bitrev_bf_out_of_place");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for &log_rows in &SUPPORTED_LOG_ROWS {
        let rows = 1usize << log_rows;
        let ring_len = stage1_ring_len(rows).unwrap();
        let mut ring_cases = (0..ring_len)
            .map(|_| HypercubeBitrevBenchCase::new(log_rows, &stream).unwrap())
            .collect::<Vec<_>>();
        let mut ring_idx = 0usize;
        group.throughput(Throughput::Bytes(ring_cases[0].bytes_per_transform()));
        group.bench_with_input(
            BenchmarkId::new("transform", format!("log_rows={}", log_rows)),
            &log_rows,
            |b: &mut Bencher<CudaMeasurement>, _| {
                b.iter(|| {
                    ring_cases[ring_idx].run_out_of_place(&stream).unwrap();
                    stream.synchronize().unwrap();
                    ring_idx += 1;
                    if ring_idx == ring_cases.len() {
                        ring_idx = 0;
                    }
                })
            },
        );
    }

    group.finish();
}

fn benchmark_in_place(c: &mut Criterion<CudaMeasurement>) {
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("hypercube_bitrev_bf_in_place");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for &log_rows in &SUPPORTED_LOG_ROWS {
        let rows = 1usize << log_rows;
        let ring_len = stage1_ring_len(rows).unwrap();
        let mut ring_cases = (0..ring_len)
            .map(|_| HypercubeBitrevBenchCase::new(log_rows, &stream).unwrap())
            .collect::<Vec<_>>();
        let mut ring_idx = 0usize;
        group.throughput(Throughput::Bytes(ring_cases[0].bytes_per_transform()));
        group.bench_with_input(
            BenchmarkId::new("transform", format!("log_rows={}", log_rows)),
            &log_rows,
            |b: &mut Bencher<CudaMeasurement>, _| {
                b.iter(|| {
                    ring_cases[ring_idx].run_in_place(&stream).unwrap();
                    stream.synchronize().unwrap();
                    ring_idx += 1;
                    if ring_idx == ring_cases.len() {
                        ring_idx = 0;
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement {});
    targets = benchmark_out_of_place, benchmark_in_place
);
criterion_main!(bench);
