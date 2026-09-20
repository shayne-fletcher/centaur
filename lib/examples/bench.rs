//! Measure sequential scalar, one-register NEON, and four-register NEON dots.

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "aarch64",
    target_os = "macos",
    target_feature = "neon"
))]
mod m4 {
    use std::hint::black_box;
    use std::process::Command;
    use std::time::Duration;
    use std::time::Instant;

    use centaur::comparison;

    const SAMPLE_COUNT: usize = 11;
    const TARGET_SAMPLE_TIME: Duration = Duration::from_millis(50);

    #[derive(Clone, Copy)]
    struct Variant {
        name: &'static str,
        function: fn(&[f32], &[f32]) -> f32,
    }

    struct Measurement {
        median_ns_per_element: f64,
        repetitions: usize,
    }

    pub fn run() {
        println!("centaur dot_f32 benchmark");
        println!("machine: {}", sysctl("machdep.cpu.brand_string"));
        println!("model: {}", sysctl("hw.model"));
        println!("architecture: {}", std::env::consts::ARCH);
        println!("rustc: {}", rustc_version());
        println!("samples: {SAMPLE_COUNT}");
        println!("target sample time: {} ms", TARGET_SAMPLE_TIME.as_millis());

        measure_fixture("L1-hot", 4096);
        measure_fixture("streaming", 4 * 1024 * 1024);
    }

    fn measure_fixture(name: &str, len: usize) {
        let lhs: Vec<_> = (0..len)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let rhs: Vec<_> = (0..len)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) / 48.0)
            .collect();
        let variants = [
            Variant {
                name: "scalar sequential",
                function: comparison::dot_f32_sequential,
            },
            Variant {
                name: "NEON one register",
                function: comparison::dot_f32_neon_one,
            },
            Variant {
                name: "NEON four registers",
                function: comparison::dot_f32_neon_four,
            },
        ];

        preflight(&lhs, &rhs, &variants);
        for variant in variants {
            for _ in 0..4 {
                black_box((variant.function)(black_box(&lhs), black_box(&rhs)));
            }
        }

        let repetitions = variants.map(|variant| calibrate(variant, &lhs, &rhs));
        let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for sample in 0..SAMPLE_COUNT {
            for offset in 0..variants.len() {
                let index = (sample + offset) % variants.len();
                samples[index].push(measure(variants[index], &lhs, &rhs, repetitions[index]));
            }
        }

        let measurements: [Measurement; 3] = std::array::from_fn(|index| Measurement {
            median_ns_per_element: median(&mut samples[index]),
            repetitions: repetitions[index],
        });
        let scalar = measurements[0].median_ns_per_element;

        println!();
        println!("fixture: {name}");
        println!("length: {len} floats per slice");
        println!("input bytes: {}", len * size_of::<f32>() * 2);
        println!("variant                 reps    median ns/element    vs scalar");
        for (variant, measurement) in variants.iter().zip(&measurements) {
            println!(
                "{:<23} {:>6} {:>20.4} {:>11.2}x",
                variant.name,
                measurement.repetitions,
                measurement.median_ns_per_element,
                scalar / measurement.median_ns_per_element,
            );
        }
        println!(
            "one-register / four-register: {:.2}x",
            measurements[1].median_ns_per_element / measurements[2].median_ns_per_element
        );
    }

    fn preflight(lhs: &[f32], rhs: &[f32], variants: &[Variant]) {
        let (oracle, bound) = comparison::oracle(lhs, rhs);
        for variant in variants {
            let result = f64::from((variant.function)(lhs, rhs));
            assert!(
                (result - oracle).abs() <= bound,
                "{} result {result} exceeds oracle {oracle} ± {bound}",
                variant.name,
            );
        }
    }

    fn calibrate(variant: Variant, lhs: &[f32], rhs: &[f32]) -> usize {
        let mut repetitions = 1_usize;
        loop {
            let elapsed = run_repetitions(variant, lhs, rhs, repetitions);
            if elapsed >= TARGET_SAMPLE_TIME {
                return repetitions;
            }
            let elapsed_ns = elapsed.as_nanos().max(1);
            let scale = (TARGET_SAMPLE_TIME.as_nanos() / elapsed_ns).clamp(2, 1024);
            repetitions = repetitions
                .checked_mul(scale as usize)
                .expect("benchmark repetition count overflowed");
        }
    }

    fn measure(variant: Variant, lhs: &[f32], rhs: &[f32], repetitions: usize) -> f64 {
        let elapsed = run_repetitions(variant, lhs, rhs, repetitions);
        elapsed.as_secs_f64() * 1e9 / repetitions as f64 / lhs.len() as f64
    }

    fn run_repetitions(variant: Variant, lhs: &[f32], rhs: &[f32], repetitions: usize) -> Duration {
        let start = Instant::now();
        for _ in 0..repetitions {
            black_box((variant.function)(black_box(lhs), black_box(rhs)));
        }
        start.elapsed()
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_unstable_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn sysctl(name: &str) -> String {
        let output = Command::new("sysctl")
            .args(["-n", name])
            .output()
            .expect("failed to run sysctl");
        String::from_utf8(output.stdout)
            .expect("sysctl returned non-UTF-8 output")
            .trim()
            .to_owned()
    }

    fn rustc_version() -> String {
        let output = Command::new("rustc")
            .arg("--version")
            .output()
            .expect("failed to run rustc --version");
        String::from_utf8(output.stdout)
            .expect("rustc returned non-UTF-8 output")
            .trim()
            .to_owned()
    }
}

#[cfg(all(
    not(feature = "scalar-only"),
    target_arch = "aarch64",
    target_os = "macos",
    target_feature = "neon"
))]
fn main() {
    m4::run();
}

#[cfg(not(all(
    not(feature = "scalar-only"),
    target_arch = "aarch64",
    target_os = "macos",
    target_feature = "neon"
)))]
fn main() {
    eprintln!("the one-versus-four benchmark requires Apple silicon macOS");
}
