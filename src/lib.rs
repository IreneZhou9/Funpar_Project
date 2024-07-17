#![feature(portable_simd)]
use std::simd::{f64x64};
use std::simd::num::SimdFloat;
use std::simd::{SupportedLaneCount};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyAny;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use numpy::{IntoPyArray, PyArray1, ToPyArray};

// dot product
#[pyfunction]
unsafe fn dot_product_par_simd(x: &PyAny, y: &PyAny) -> PyResult<f64> {
    let x_array: &PyArray1<f64> = x.extract()?;
    let y_array: &PyArray1<f64> = y.extract()?;

    const PAR_CHUNK_SIZE: usize = 16_384;

    Ok(x_array.as_slice()?.par_chunks(PAR_CHUNK_SIZE)
        .zip(y_array.as_slice()?.par_chunks(PAR_CHUNK_SIZE))
        .map(|(chunk_x, chunk_y)| dot(chunk_x, chunk_y))
        .sum())
}

fn dot(xs: &[f64], ys: &[f64]) -> f64 {
    const CHUNK_SIZE: usize = 64;
    let packed_x = xs.chunks_exact(CHUNK_SIZE).map(|chunk| f64x64::from_slice(chunk));
    let packed_y = ys.chunks_exact(CHUNK_SIZE).map(|chunk| f64x64::from_slice(chunk));

    packed_x.zip(packed_y).map(|(xi, yi)| xi * yi).sum::<f64x64>().reduce_sum()
}

// argsort
#[pyfunction]
fn argsort(array: Vec<i32>) -> PyResult<Vec<usize>> {
    let mut indices: Vec<usize> = (0..array.len()).collect();
    indices.par_sort_by(|&i, &j| array[i].cmp(&array[j]));
    Ok(indices)
}

// convolve
#[pyfunction]
/// Convolve two 1-dimensional slices using direct summation
fn convolve(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let conv_len = n + m - 1;

    // Use Rayon to parallelize the outer loop
    let result: Vec<f64> = (0..conv_len)
        .into_par_iter()
        .map(|i| {
            let mut sum = 0.0;
            let start = if i >= m { i - m + 1 } else { 0 };
            let end = if i < n { i + 1 } else { n };

            for j in start..end {
                sum += a[j] * b[i - j];
            }

            sum
        })
        .collect();

    Ok(result)
}

#[pyfunction]
/// Convolve two 1-dimensional slices using direct summation
fn convolve_exp(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = a.len();
    let m = b.len();
    let conv_len = n + m - 1;

    // Create a thread pool
    let pool = ThreadPoolBuilder::new().build().unwrap();

    let result: Vec<f64> = pool.install(|| {
        (0..conv_len)
            .into_par_iter()
            .map(|i| {
                let start = if i >= m { i - m + 1 } else { 0 };
                let end = if i < n { i + 1 } else { n };
                (start..end)
                    .map(|j| a[j] * b[i - j])
                    .sum::<f64>()
            })
            .collect()
    });

    Ok(result)
}

/// A Python module implemented in Rust.
#[pymodule]
fn dotpro(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product_par_simd, m)?)?;
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    m.add_function(wrap_pyfunction!(convolve_exp, m)?)?;
    Ok(())
}
