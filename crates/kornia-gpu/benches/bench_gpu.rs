//! Benchmarks: GPU vs CPU for cast_and_scale, warp_perspective, gray_from_rgb,
//! and the full BEV pipeline.
//!
//! Run with:
//!   cargo bench -p kornia-gpu --bench bench_gpu
//!
//! Two timing modes per GPU operation:
//!   gpu_compute  — kernel only (data already in VRAM, result stays in VRAM)
//!   gpu_e2e      — upload + kernel + download (true wall-clock cost)
//!
//! The compute time shows raw kernel speedup.
//! The e2e time shows real-world speedup when data must cross the PCIe bus.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_gpu::{kernels, pipeline::GpuPipeline, GpuAllocator, ImageExt};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_tensor::CpuAllocator;

const SIZES: &[(usize, usize)] = &[
    (720, 1280),
    (1080, 1920),
    (2160, 3840),
];

fn make_cpu_image(h: usize, w: usize) -> Image<f32, 3, CpuAllocator> {
    let data: Vec<f32> = (0..h * w * 3).map(|i| (i % 256) as f32 / 255.0).collect();
    Image::new(ImageSize { height: h, width: w }, data, CpuAllocator).unwrap()
}

fn bev_homography() -> [f32; 9] {
    [1.2, 0.1, -100.0, -0.05, 1.1, -80.0, 0.0001, 0.0002, 1.0]
}

// ---------------------------------------------------------------------------
// cast_and_scale
// ---------------------------------------------------------------------------

fn bench_cast_and_scale(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let mut group = c.benchmark_group("cast_and_scale");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // CPU (rayon)
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| cpu_img.clone().cast_and_scale::<f32>(1.0 / 255.0).unwrap())
        });

        // GPU compute only — data pre-uploaded, result stays in VRAM
        let gpu_img = cpu_img.to_gpu(&gpu).unwrap();
        group.bench_with_input(BenchmarkId::new("gpu_compute", &label), &(), |b, _| {
            b.iter(|| kernels::cast_and_scale(&gpu_img, 1.0 / 255.0).unwrap())
        });

        // GPU end-to-end — includes upload + kernel + download
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                kernels::cast_and_scale(&cpu_img.to_gpu(&gpu).unwrap(), 1.0 / 255.0)
                    .unwrap()
                    .to_cpu()
                    .unwrap()
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// warp_perspective
// ---------------------------------------------------------------------------

fn bench_warp_perspective(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let m = bev_homography();
    let mut group = c.benchmark_group("warp_perspective");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // CPU (kornia-imgproc rayon reference)
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            let mut dst = Image::<f32, 3, _>::from_size_val(
                ImageSize { height: h, width: w }, 0.0, CpuAllocator,
            ).unwrap();
            b.iter(|| kornia_imgproc::warp::warp_perspective(
                &cpu_img, &mut dst, &m, InterpolationMode::Bilinear,
            ).unwrap())
        });

        // GPU compute only
        let gpu_img = cpu_img.to_gpu(&gpu).unwrap();
        group.bench_with_input(BenchmarkId::new("gpu_compute", &label), &(), |b, _| {
            b.iter(|| kernels::warp_perspective(&gpu_img, (h, w), &m).unwrap())
        });

        // GPU end-to-end
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                kernels::warp_perspective(&cpu_img.to_gpu(&gpu).unwrap(), (h, w), &m)
                    .unwrap()
                    .to_cpu()
                    .unwrap()
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// gray_from_rgb
// ---------------------------------------------------------------------------

fn bench_gray_from_rgb(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let mut group = c.benchmark_group("gray_from_rgb");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            let mut dst = Image::<f32, 1, _>::from_size_val(
                ImageSize { height: h, width: w }, 0.0, CpuAllocator,
            ).unwrap();
            b.iter(|| kornia_imgproc::color::gray_from_rgb(&cpu_img, &mut dst).unwrap())
        });

        // GPU compute only
        let gpu_img = cpu_img.to_gpu(&gpu).unwrap();
        group.bench_with_input(BenchmarkId::new("gpu_compute", &label), &(), |b, _| {
            b.iter(|| kernels::gray_from_rgb(&gpu_img).unwrap())
        });

        // GPU end-to-end
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                kernels::gray_from_rgb(&cpu_img.to_gpu(&gpu).unwrap())
                    .unwrap()
                    .to_cpu()
                    .unwrap()
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Full BEV pipeline
// ---------------------------------------------------------------------------

fn bench_bev_pipeline(c: &mut Criterion) {
    let gpu = GpuAllocator::new();
    let m = bev_homography();
    let mut group = c.benchmark_group("bev_pipeline");

    for &(h, w) in SIZES {
        let label = format!("{}x{}", w, h);
        let cpu_img = make_cpu_image(h, w);

        // CPU: cast → warp → gray (3 separate rayon passes)
        group.bench_with_input(BenchmarkId::new("cpu", &label), &(), |b, _| {
            b.iter(|| {
                let scaled = cpu_img.clone().cast_and_scale::<f32>(1.0 / 255.0).unwrap();
                let mut warped = Image::<f32, 3, _>::from_size_val(
                    ImageSize { height: h, width: w }, 0.0, CpuAllocator,
                ).unwrap();
                kornia_imgproc::warp::warp_perspective(
                    &scaled, &mut warped, &m, InterpolationMode::Bilinear,
                ).unwrap();
                let mut gray = Image::<f32, 1, _>::from_size_val(
                    ImageSize { height: h, width: w }, 0.0, CpuAllocator,
                ).unwrap();
                kornia_imgproc::color::gray_from_rgb(&warped, &mut gray).unwrap();
                gray
            })
        });

        // GPU pipeline: upload once → 3 kernels → download once
        group.bench_with_input(BenchmarkId::new("gpu_e2e", &label), &(), |b, _| {
            b.iter(|| {
                GpuPipeline::new(&gpu)
                    .upload(&cpu_img).unwrap()
                    .cast_and_scale(1.0 / 255.0).unwrap()
                    .warp_perspective((h, w), &m).unwrap()
                    .gray_from_rgb().unwrap()
                    .download().unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cast_and_scale,
    bench_warp_perspective,
    bench_gray_from_rgb,
    bench_bev_pipeline,
);
criterion_main!(benches);