#![feature(portable_simd)]

use std::simd::{f64x16, f64x2, f64x32, f64x4, f64x64, f64x8};
use std::simd::num::SimdFloat;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyList, PyAny};
use rayon::prelude::*;
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, ToPyArray};
use ndarray::{Array2, Axis};

// Testing...
#[pyfunction]
// fn dot_product(x: Vec<i32>, y: Vec<i32>) -> PyResult<i32> {
//     let result = x.par_iter().zip(y.par_iter()).map(|(xi, yi)| xi * yi).sum();
//     Ok(result)
// }
fn dot_product(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    Ok(x.par_iter().zip(y.par_iter()).map(|(xi, yi)| xi * yi).sum())
}

#[pyfunction]
fn dot_product_any(py: Python, x: &PyAny, y: &PyAny) -> PyResult<f64> {
    let x_array: &PyArray1<f64> = x.extract()?;
    let y_array: &PyArray1<f64> = y.extract()?;

    let x_vec: Vec<f64> = x_array.to_vec()?;
    let y_vec: Vec<f64> = y_array.to_vec()?;

    let result: f64 = x_vec.par_iter().zip(y_vec.par_iter()).map(|(xi, yi)| xi * yi).sum();
    Ok(result)
}

#[pyfunction]
unsafe fn dot_product_par_simd(x: &PyAny, y: &PyAny) -> PyResult<f64> {
    let x_array: &PyArray1<f64> = x.extract()?;
    let y_array: &PyArray1<f64> = y.extract()?;

    // let x_vec: Vec<f64> = x_array.to_vec()?;
    // let y_vec: Vec<f64> = y_array.to_vec()?;

    const PAR_CHUNK_SIZE: usize = 16_384;

    // Ok(x_vec.par_chunks(PAR_CHUNK_SIZE).zip(y_vec.par_chunks(PAR_CHUNK_SIZE)).map(|(chunk_x, chunk_y)| dot(chunk_x, chunk_y)).sum())

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

fn convert_pylist_to_array(py: Python, obj: &PyList) -> PyResult<Array2<f64>> {
    let rows = obj.len();
    let cols = obj.get_item(0)?.downcast::<PyList>()?.len();
    let mut matrix = Vec::new();
    for row in obj.iter() {
        let row_list = row.downcast::<PyList>()?;
        for elem in row_list.iter() {
            let val: f64 = elem.extract()?;
            matrix.push(val);
        }
    }
    Ok(Array2::from_shape_vec((rows, cols), matrix).unwrap())
}

#[pyfunction]
fn matrix_mul_optimized_parallel(py: Python, mat1: &PyAny, mat2: &PyAny) -> PyResult<PyObject> {
    let mat1_pylist = mat1.downcast::<PyList>()?;
    let mat2_pylist = mat2.downcast::<PyList>()?;
    let mat1_array = convert_pylist_to_array(py, mat1_pylist)?;
    let mat2_array = convert_pylist_to_array(py, mat2_pylist)?;

    let mat1_rows = mat1_array.nrows();
    let mat2_cols = mat2_array.ncols();
    let mut result = Array2::<f64>::zeros((mat1_rows, mat2_cols));

    result.axis_iter_mut(Axis(0)).enumerate().par_bridge().for_each(|(i, mut row)| {
        for j in 0..mat2_cols {
            let mut sum = 0.0;
            for k in 0..mat2_array.nrows() {
                sum += mat1_array[[i, k]] * mat2_array[[k, j]];
            }
            row[j] = sum;
        }
    });

    let result_pylist = result.map(|&x| x.to_object(py)).into_pyarray(py).to_object(py);
    Ok(result_pylist)
}
#[pyfunction]
fn par_argsort(xs: Vec<i32>) -> PyResult<Vec<usize>> {
    type Candidate = (i32, usize);

    fn parallel_merge(left: &[Candidate], right: &[Candidate], combined: &mut Vec<Candidate>) {
        let total_len = left.len() + right.len();
        combined.reserve(total_len);

        let left_mid = left.len() / 2;
        let right_mid = match right.binary_search_by(|probe| probe.0.cmp(&left[left_mid].0)) {
            Ok(pos) | Err(pos) => pos,
        };

        let (left_low, left_high) = left.split_at(left_mid);
        let (right_low, right_high) = right.split_at(right_mid);

        let (mut combined_low, mut combined_high) = combined.split_at_mut(total_len - left_high.len() - right_high.len());

        rayon::join(
            || parallel_merge(left_low, right_low, &mut combined_low.to_vec()),
            || parallel_merge(left_high, right_high, &mut combined_high.to_vec()),
        );

        let (mut i, mut j) = (0, 0);
        while i < left_low.len() && j < right_low.len() {
            if left_low[i].0 <= right_low[j].0 {
                combined.push(left_low[i]);
                i += 1;
            } else {
                combined.push(right_low[j]);
                j += 1;
            }
        }

        combined.extend_from_slice(&left_low[i..]);
        combined.extend_from_slice(&right_low[j..]);
    }

    fn par_argsort_internal(xs: &[i32]) -> Vec<usize> {
        if xs.len() <= 1 {
            return (0..xs.len()).collect();
        }

        let mid = xs.len() / 2;
        let (left, right) = xs.split_at(mid);

        let (left_sorted, right_sorted): (Vec<usize>, Vec<usize>) = rayon::join(
            || par_argsort_internal(left),
            || par_argsort_internal(right),
        );

        let left_candidates: Vec<Candidate> = left_sorted.iter().map(|&i| (left[i], i)).collect();
        let right_candidates: Vec<Candidate> = right_sorted.iter().map(|&i| (right[i], i + mid)).collect();

        let mut merged_candidates = Vec::with_capacity(xs.len());
        parallel_merge(&left_candidates, &right_candidates, &mut merged_candidates);

        merged_candidates.into_iter().map(|c| c.1).collect()
    }

    Ok(par_argsort_internal(&xs))
}

/// A Python module implemented in Rust.
#[pymodule]
fn dotpro(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_mul_optimized_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product_any, m)?)?;
    m.add_function(wrap_pyfunction!(par_argsort, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product_par_simd, m)?)?;
    Ok(())
}
