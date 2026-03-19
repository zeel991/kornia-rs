//! GPU-accelerated image processing for kornia-rs.
//!
//! This crate provides GPU implementations of kornia-imgproc operations using
//! [CubeCL](https://github.com/tracel-ai/cubecl) as the compute backend.
//! CubeCL compiles the same Rust kernel code to Vulkan (wgpu), CUDA, ROCm,
//! and Metal — so the same kernels run on any GPU without code changes.
//!
//! # Architecture
//!
//! ```text
//! kornia-gpu
//! ├── allocator   GpuAllocator (implements TensorAllocator) + GpuMemory<T>
//! ├── image       GpuImage<T, C> + to_gpu() / to_cpu() transfer API
//! ├── kernels
//! │   ├── cast    cast_and_scale — GPU version of Image::cast_and_scale
//! │   ├── warp    warp_perspective — GPU version of imgproc::warp_perspective
//! │   └── color   gray_from_rgb — GPU version of imgproc::gray_from_rgb
//! └── pipeline    GpuPipeline — zero-copy kernel chaining
//! ```
//!
//! # Integration with kornia-imgproc
//!
//! The intended usage from kornia-imgproc (behind a `cubecl` feature flag):
//!
//! ```rust,ignore
//! // In kornia-imgproc/src/warp/perspective.rs:
//! pub fn warp_perspective<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
//!     src: &Image<f32, C, A1>,
//!     dst: &mut Image<f32, C, A2>,
//!     m: &[f32; 9],
//!     interpolation: InterpolationMode,
//! ) -> Result<(), ImageError> {
//!     // GPU dispatch: if src is backed by GpuAllocator, use GPU kernel
//!     #[cfg(feature = "cubecl")]
//!     if let (Some(gpu_src), Some(_gpu_dst)) = (
//!         src.as_gpu_image(),
//!         dst.as_gpu_image_mut(),
//!     ) {
//!         return kornia_gpu::kernels::warp_perspective(gpu_src, dst_size, m)
//!             .map(|_| ())
//!             .map_err(Into::into);
//!     }
//!     // CPU fallback (unchanged)
//!     // ...
//! }
//! ```
//!
//! # Quick start
//!
//! ```rust,ignore
//! use kornia_gpu::{GpuAllocator, pipeline::GpuPipeline};
//! use kornia_image::{Image, ImageSize};
//! use kornia_tensor::CpuAllocator;
//!
//! let cpu_img = Image::<f32, 3, _>::from_size_val(
//!     ImageSize { width: 1920, height: 1080 },
//!     0.5,
//!     CpuAllocator,
//! )?;
//!
//! let homography = [1.2, 0.1, -100.0, -0.05, 1.1, -80.0, 0.0001, 0.0002, 1.0];
//!
//! let gpu = GpuAllocator::new();
//! let result = GpuPipeline::new(&gpu)
//!     .upload(&cpu_img)?
//!     .cast_and_scale(1.0 / 255.0)?
//!     .warp_perspective((1080, 1920), &homography)?
//!     .gray_from_rgb()?
//!     .download()?;
//! ```

pub mod allocator;
pub mod error;
pub mod image;
pub mod kernels;
pub mod pipeline;
pub mod pool;

pub use allocator::GpuAllocator;
pub use error::GpuError;
pub use image::{GpuImage, ImageExt};
pub use pool::GpuImagePool;

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageSize};
    use kornia_tensor::CpuAllocator;

    use crate::{
        kernels,
        pipeline::GpuPipeline,
        GpuAllocator, GpuImage, ImageExt,
    };

    fn gpu() -> GpuAllocator {
        GpuAllocator::new()
    }

    fn rgb_image(h: usize, w: usize) -> Image<f32, 3, CpuAllocator> {
        let data: Vec<f32> = (0..h * w * 3).map(|i| (i % 256) as f32).collect();
        Image::new(ImageSize { height: h, width: w }, data, CpuAllocator).unwrap()
    }

    // -----------------------------------------------------------------------
    // Transfer round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_transfer_roundtrip() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let gpu_img = src.to_gpu(&gpu).unwrap();
        let back = gpu_img.to_cpu().unwrap();
        assert_eq!(src.as_slice(), back.as_slice());
        assert_eq!(src.size(), back.size());
    }

    // -----------------------------------------------------------------------
    // cast_and_scale
    // -----------------------------------------------------------------------

    #[test]
    fn test_cast_and_scale_exact() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let gpu_src = src.to_gpu(&gpu).unwrap();

        let gpu_out = kernels::cast_and_scale(&gpu_src, 1.0 / 255.0).unwrap();
        let result = gpu_out.to_cpu().unwrap();

        // CPU reference
        let expected: Vec<f32> = src.as_slice().iter().map(|&v| v / 255.0).collect();

        for (a, b) in result.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "cast_and_scale mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_cast_and_scale_zero() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 0.0, CpuAllocator,
        ).unwrap();
        let result = kernels::cast_and_scale(&src.to_gpu(&gpu).unwrap(), 1.0 / 255.0)
            .unwrap().to_cpu().unwrap();
        assert!(result.as_slice().iter().all(|&v| v == 0.0));
    }

    // -----------------------------------------------------------------------
    // gray_from_rgb
    // -----------------------------------------------------------------------

    #[test]
    fn test_gray_from_rgb_black() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 0.0, CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        assert!(result.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gray_from_rgb_white() {
        let gpu = gpu();
        let src = Image::<f32, 3, _>::from_size_val(
            ImageSize { height: 4, width: 4 }, 1.0, CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        // 0.299 + 0.587 + 0.114 = 1.0
        for &v in result.as_slice() {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_gray_from_rgb_matches_weights() {
        // Single pixel: R=1.0, G=0.0, B=0.0 → gray = 0.299
        let gpu = gpu();
        let src = Image::<f32, 3, _>::new(
            ImageSize { height: 1, width: 1 },
            vec![1.0, 0.0, 0.0],
            CpuAllocator,
        ).unwrap();
        let result = kernels::gray_from_rgb(&src.to_gpu(&gpu).unwrap())
            .unwrap().to_cpu().unwrap();
        assert!((result.as_slice()[0] - 0.299).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // warp_perspective
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_perspective_identity_shape() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (8, 8), &identity)
            .unwrap().to_cpu().unwrap();
        assert_eq!(result.size(), src.size());
        assert_eq!(result.num_channels(), 3);
    }

    #[test]
    fn test_warp_perspective_identity_values() {
        // Interior pixels should match under identity transform
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (8, 8), &identity)
            .unwrap().to_cpu().unwrap();

        // Interior 4x4 pixels should match src exactly (no border effects)
        for row in 1..6usize {
            for col in 1..6usize {
                for ch in 0..3usize {
                    let expected = src.get([row, col, ch]).unwrap();
                    let actual = result.get([row, col, ch]).unwrap();
                    assert!((expected - actual).abs() < 1e-4,
                        "pixel ({row},{col},{ch}): expected {expected}, got {actual}");
                }
            }
        }
    }

    #[test]
    fn test_warp_perspective_output_size() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (4, 6), &m)
            .unwrap().to_cpu().unwrap();
        assert_eq!(result.height(), 4);
        assert_eq!(result.width(), 6);
    }

    #[test]
    fn test_warp_perspective_singular_homography() {
        let gpu = gpu();
        let src = rgb_image(4, 4);
        let singular = [0.0f32; 9]; // det = 0
        let err = kernels::warp_perspective(&src.to_gpu(&gpu).unwrap(), (4, 4), &singular);
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // GpuPipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_cast_warp() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = GpuPipeline::new(&gpu)
            .upload(&src).unwrap()
            .cast_and_scale(1.0 / 255.0).unwrap()
            .warp_perspective((8, 8), &m).unwrap()
            .download().unwrap();

        assert_eq!(result.size(), src.size());
        assert_eq!(result.num_channels(), 3);
    }

    #[test]
    fn test_pipeline_full_bev() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = GpuPipeline::new(&gpu)
            .upload(&src).unwrap()
            .cast_and_scale(1.0 / 255.0).unwrap()
            .warp_perspective((8, 8), &m).unwrap()
            .gray_from_rgb().unwrap()
            .download().unwrap();

        assert_eq!(result.height(), 8);
        assert_eq!(result.width(), 8);
        assert_eq!(result.num_channels(), 1);
    }

    /// Correctness test at full 1080p resolution.
    /// This is the actual target resolution for the BEV pipeline.
    /// Also serves as evidence the kernel handles large workloads correctly.
    #[test]
    fn test_warp_perspective_1080p_correctness() {
        let gpu = gpu();
        let src = rgb_image(1080, 1920);
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let result = kernels::warp_perspective(
            &src.to_gpu(&gpu).unwrap(), (1080, 1920), &identity,
        ).unwrap().to_cpu().unwrap();

        assert_eq!(result.size(), src.size());

        // Check centre pixel — far from borders, bilinear on integer coords is exact
        let row = 540usize;
        let col = 960usize;
        for ch in 0..3usize {
            let expected = src.get([row, col, ch]).unwrap();
            let actual   = result.get([row, col, ch]).unwrap();
            assert!(
                (expected - actual).abs() < 1e-4,
                "1080p pixel ({row},{col},{ch}): expected {expected}, got {actual}"
            );
        }
    }
    // -----------------------------------------------------------------------
    // GpuImagePool — persistent VRAM reuse
    // -----------------------------------------------------------------------

    #[test]
    fn test_pool_acquire_release() {
        let gpu = gpu();
        let pool = crate::pool::GpuImagePool::<f32, 3>::new(2, 4, 4, &gpu).unwrap();
        assert_eq!(pool.available_count(), 2);

        let buf = pool.acquire().unwrap();
        assert_eq!(pool.available_count(), 1);

        pool.release(buf);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_pool_exhaustion() {
        let gpu = gpu();
        let pool = crate::pool::GpuImagePool::<f32, 3>::new(1, 4, 4, &gpu).unwrap();
        let _buf = pool.acquire().unwrap();
        let result = pool.acquire();
        assert!(matches!(result, Err(crate::error::GpuError::PoolExhausted)));
    }

    #[test]
    fn test_cast_and_scale_into_reuse() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let gpu_src = src.to_gpu(&gpu).unwrap();

        // Pre-allocate output buffer — zero per-call allocation
        let dst = GpuImage::<f32, 3>::empty(8, 8, &gpu);
        kernels::cast_and_scale_into(&gpu_src, &dst, 1.0 / 255.0).unwrap();
        let result = dst.to_cpu().unwrap();

        let expected: Vec<f32> = src.as_slice().iter().map(|&v| v / 255.0).collect();
        for (a, b) in result.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_warp_perspective_into_reuse() {
        let gpu = gpu();
        let src = rgb_image(8, 8);
        let gpu_src = src.to_gpu(&gpu).unwrap();
        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Pre-allocate output buffer
        let dst = GpuImage::<f32, 3>::empty(8, 8, &gpu);
        kernels::warp_perspective_into(&gpu_src, &dst, &identity).unwrap();
        let result = dst.to_cpu().unwrap();
        assert_eq!(result.size(), src.size());
    }

    #[test]
    fn test_pool_pipeline_no_allocation() {
        // Simulates a 3-frame pipeline where buffers are reused every frame.
        // Pool allocates twice at startup, then zero allocations per frame.
        let gpu = gpu();
        let pool_warp = crate::pool::GpuImagePool::<f32, 3>::new(1, 8, 8, &gpu).unwrap();
        let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        for _ in 0..3 {
            let src = rgb_image(8, 8).to_gpu(&gpu).unwrap();
            let warp_buf = pool_warp.acquire().unwrap();
            kernels::warp_perspective_into(&src, &warp_buf, &m).unwrap();
            let _result = warp_buf.to_cpu().unwrap();
            pool_warp.release(warp_buf); // return for next frame
        }
        assert_eq!(pool_warp.available_count(), 1);
    }

}