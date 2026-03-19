//! Persistent VRAM buffer pool for reusable GPU image allocations.
//!
//! # Problem
//!
//! Without pooling, every kernel call allocates a new GPU buffer:
//!   cast_and_scale → GpuImage::empty() → client.create() → new VRAM handle
//!
//! In a real-time pipeline running at 30fps, this means 30 allocations/second
//! per kernel stage. CubeCL manages a pool internally but the handle churn
//! still has overhead. More importantly, it prevents the "persistent VRAM
//! buffer" architecture where the same memory region is reused every frame.
//!
//! # Solution
//!
//! `GpuImagePool` pre-allocates a fixed set of `GpuImage` buffers at startup.
//! Callers `acquire()` a buffer, use it, then `release()` it back.
//!
//! In the BEV pipeline this means:
//!   - Frame 1: acquire buf_a, run warp into buf_a, acquire buf_b, run gray into buf_b
//!   - Frame 2: same buf_a and buf_b reused — zero new VRAM allocations
//!
//! # Usage
//!
//! ```rust,ignore
//! use kornia_gpu::pool::GpuImagePool;
//!
//! let pool = GpuImagePool::<f32, 3>::new(2, 1080, 1920, &gpu)?;
//!
//! // In the frame loop:
//! let buf = pool.acquire()?;
//! kernels::warp_perspective_into(&src, &buf, (1080, 1920), &homography)?;
//! let result = buf.to_cpu()?;
//! pool.release(buf);
//! ```

use std::sync::{Arc, Mutex};

use crate::allocator::GpuAllocator;
use crate::error::GpuError;
use crate::image::GpuImage;

/// A pool of pre-allocated GPU image buffers.
///
/// All buffers in the pool share the same dimensions and channel count.
/// This is appropriate for a fixed-resolution pipeline (e.g. all frames
/// are 1920×1080 RGB).
pub struct GpuImagePool<T: bytemuck::Pod + Send + Sync, const C: usize> {
    /// Available (not currently in use) buffers.
    available: Arc<Mutex<Vec<GpuImage<T, C>>>>,
}

impl<T: bytemuck::Pod + Send + Sync, const C: usize> GpuImagePool<T, C> {
    /// Create a pool with `capacity` pre-allocated buffers of size `height × width`.
    ///
    /// All VRAM allocations happen here, at startup — not per-frame.
    pub fn new(
        capacity: usize,
        height: usize,
        width: usize,
        alloc: &GpuAllocator,
    ) -> Result<Self, GpuError> {
        let mut buffers = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffers.push(GpuImage::<T, C>::empty(height, width, alloc));
        }
        Ok(Self {
            available: Arc::new(Mutex::new(buffers)),
        })
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns `Err(GpuError::PoolExhausted)` if no buffers are available.
    /// The caller must `release()` the buffer when done or it is permanently removed.
    pub fn acquire(&self) -> Result<GpuImage<T, C>, GpuError> {
        self.available
            .lock()
            .unwrap()
            .pop()
            .ok_or(GpuError::PoolExhausted)
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, buf: GpuImage<T, C>) {
        self.available.lock().unwrap().push(buf);
    }

    /// Number of buffers currently available (not acquired).
    pub fn available_count(&self) -> usize {
        self.available.lock().unwrap().len()
    }
}