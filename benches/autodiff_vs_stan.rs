// Benchmark: Autodiff vs Stan (BridgeStan) implementation
// Compares the performance of the pure Rust autodiff implementation vs the Stan-based one

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array1, Array2};

#[cfg(feature = "autodiff-backend")]
use farseer::core::prophet_autodiff::ProphetModel;
#[cfg(feature = "autodiff-backend")]
use farseer::core::prophet_optimizer::{optimize_prophet, initialize_params, OptimizationConfig};

fn create_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Array2<f64>) {
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut x = Array2::zeros((n, 4)); // 4 seasonality features (2 fourier terms)
    
    for i in 0..n {
        t[i] = i as f64 / n as f64;
        
        // Sinusoidal pattern for Fourier features
        let period_val = 2.0 * std::f64::consts::PI * t[i] * 2.0;
        x[[i, 0]] = period_val.cos();
        x[[i, 1]] = period_val.sin();
        x[[i, 2]] = (period_val * 2.0).cos();
        x[[i, 3]] = (period_val * 2.0).sin();
        
        // Generate y with trend + seasonality + noise
        y[i] = 10.0 + 0.5 * t[i] + 0.3 * x[[i, 0]] + 0.2 * x[[i, 1]] + 0.05 * (i as f64).sin();
    }
    
    (t, y, x)
}

#[cfg(feature = "autodiff-backend")]
fn bench_autodiff_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_optimization");
    
    for n in [50, 100, 200].iter() {
        let (t, y, x) = create_test_data(*n);
        
        let model = ProphetModel {
            n: *n,
            k: 4,
            s: 3, // 3 changepoints
            t: Array1::from_vec(t.clone()),
            y: Array1::from_vec(y.clone()),
            cap: Array1::ones(*n) * 100.0,
            x,
            t_change: Array1::from_vec(vec![0.25, 0.5, 0.75]),
            sigmas: Array1::ones(4),
            tau: 0.05,
            s_a: Array1::ones(4),
            s_m: Array1::zeros(4),
            weights: Array1::ones(*n),
            trend_indicator: 0, // Linear
        };
        
        let y_array = Array1::from_vec(y);
        let y_mean = y_array.mean().unwrap();
        let y_std = y_array.std(0.0);
        let init_params = initialize_params(&model, y_mean, y_std);
        
        let config = OptimizationConfig {
            max_iters: 100,
            tolerance: 1e-6,
            history_size: 5,
        };
        
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                optimize_prophet(
                    black_box(model.clone()),
                    black_box(init_params.clone()),
                    black_box(config.clone()),
                )
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "autodiff-backend")]
fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");
    
    for n in [50, 100, 200].iter() {
        let (t, y, x) = create_test_data(*n);
        
        let model = ProphetModel {
            n: *n,
            k: 4,
            s: 3,
            t: Array1::from_vec(t),
            y: Array1::from_vec(y.clone()),
            cap: Array1::ones(*n) * 100.0,
            x,
            t_change: Array1::from_vec(vec![0.25, 0.5, 0.75]),
            sigmas: Array1::ones(4),
            tau: 0.05,
            s_a: Array1::ones(4),
            s_m: Array1::zeros(4),
            weights: Array1::ones(*n),
            trend_indicator: 0,
        };
        
        let y_array = Array1::from_vec(y);
        let y_mean = y_array.mean().unwrap();
        let y_std = y_array.std(0.0);
        let params = initialize_params(&model, y_mean, y_std);
        
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                model.gradient(black_box(&params))
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "autodiff-backend")]
fn bench_neg_log_prob(c: &mut Criterion) {
    let mut group = c.benchmark_group("neg_log_prob");
    
    for n in [50, 100, 200].iter() {
        let (t, y, x) = create_test_data(*n);
        
        let model = ProphetModel {
            n: *n,
            k: 4,
            s: 3,
            t: Array1::from_vec(t),
            y: Array1::from_vec(y.clone()),
            cap: Array1::ones(*n) * 100.0,
            x,
            t_change: Array1::from_vec(vec![0.25, 0.5, 0.75]),
            sigmas: Array1::ones(4),
            tau: 0.05,
            s_a: Array1::ones(4),
            s_m: Array1::zeros(4),
            weights: Array1::ones(*n),
            trend_indicator: 0,
        };
        
        let y_array = Array1::from_vec(y);
        let y_mean = y_array.mean().unwrap();
        let y_std = y_array.std(0.0);
        let params = initialize_params(&model, y_mean, y_std);
        
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                model.neg_log_prob(black_box(&params))
            });
        });
    }
    
    group.finish();
}

#[cfg(not(feature = "autodiff-backend"))]
fn bench_autodiff_optimization(_c: &mut Criterion) {
    println!("Autodiff backend not enabled. Compile with --features autodiff-backend");
}

#[cfg(not(feature = "autodiff-backend"))]
fn bench_gradient_computation(_c: &mut Criterion) {}

#[cfg(not(feature = "autodiff-backend"))]
fn bench_neg_log_prob(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_neg_log_prob,
    bench_gradient_computation,
    bench_autodiff_optimization
);
criterion_main!(benches);
