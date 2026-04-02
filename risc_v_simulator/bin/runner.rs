use risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;
use risc_v_simulator::runner::run_simple_with_entry_point_and_non_determimism_source;
use risc_v_simulator::sim::SimulatorConfig;

fn parse_args() -> (String, Option<String>, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut bin_path: Option<String> = None;
    let mut input_file: Option<String> = None;
    let mut cycles: usize = 5_000_000_000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bin" => {
                i += 1;
                bin_path = Some(args.get(i).expect("--bin requires a path argument").clone());
            }
            "--input-file" => {
                i += 1;
                input_file = Some(
                    args.get(i)
                        .expect("--input-file requires a path argument")
                        .clone(),
                );
            }
            "--cycles" => {
                i += 1;
                cycles = args
                    .get(i)
                    .expect("--cycles requires a numeric argument")
                    .parse()
                    .expect("--cycles must be a valid integer");
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                eprintln!("Usage: runner --bin <path> [--input-file <path>] [--cycles <N>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let bin_path = bin_path.unwrap_or_else(|| {
        eprintln!("Error: --bin <path> is required");
        eprintln!("Usage: runner --bin <path> [--input-file <path>] [--cycles <N>]");
        std::process::exit(1);
    });

    (bin_path, input_file, cycles)
}

fn build_oracle(input_file: &str) -> Vec<u32> {
    let bytes = std::fs::read(input_file)
        .unwrap_or_else(|e| panic!("Failed to read input file '{}': {}", input_file, e));
    let num_bytes = bytes.len() as u32;
    let num_words = (bytes.len() + 3) / 4;
    let mut oracle = Vec::with_capacity(1 + num_words);
    oracle.push(num_bytes);
    for chunk in bytes.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        oracle.push(u32::from_le_bytes(word));
    }
    oracle
}

pub fn main() {
    let (bin_path, input_file, cycles) = parse_args();

    println!("ZK RISC-V simulator is starting");
    println!("  bin:        {}", bin_path);
    if let Some(ref f) = input_file {
        println!("  input-file: {}", f);
    }
    println!("  cycles:     {}", cycles);

    let reads = match input_file {
        Some(ref path) => build_oracle(path),
        None => vec![],
    };

    let source = QuasiUARTSource::new_with_reads(reads);

    let mut config = SimulatorConfig::simple(&bin_path);
    config.entry_point = 0;
    config.cycles = cycles;
    config.diagnostics = None;

    let output = run_simple_with_entry_point_and_non_determimism_source(config, source);
    println!("Reached end: {}", output.reached_end);
    println!(
        "Cycles: {}  Time: {:.2}s  Freq: {} cycles/s",
        output.measurements.time.exec_cycles,
        output.measurements.time.exec_time.as_secs_f64(),
        output.measurements.time.freq(),
    );
    println!("Registers a0-a7: {:?}", &output.state.registers[10..18]);
}
