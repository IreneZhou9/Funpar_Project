use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use nalgebra::DMatrix;
use numpy::{PyArray1, ToPyArray};

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
fn matrix_multiply(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> PyResult<Vec<Vec<i32>>> {
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_cols = b[0].len();

    let a_matrix = DMatrix::from_iterator(a_rows, a_cols, a.into_iter().flatten());
    let b_matrix = DMatrix::from_iterator(a_cols, b_cols, b.into_iter().flatten());

    let result_matrix = a_matrix * b_matrix;

    let result: Vec<Vec<i32>> = result_matrix
        .as_slice()
        .chunks(b_cols)
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(result)
}

#[pyfunction]
fn par_argsort(xs: &[i32]) -> Vec<usize> {
    type Candidate = (i32, usize);

    fn merge(left: &[Candidate], right: &[Candidate]) -> Vec<Candidate> {
        let mut combined = Vec::with_capacity(left.len() + right.len());
        let (mut i, mut j) = (0, 0);

        while i < left.len() && j < right.len() {
            if left[i].0 <= right[j].0 {
                combined.push(left[i]);
                i += 1;
            } else {
                combined.push(right[j]);
                j += 1;
            }
        }

        if i < left.len() {
            combined.extend_from_slice(&left[i..]);
        }

        if j < right.len() {
            combined.extend_from_slice(&right[j..]);
        }

        combined
    }

    if xs.len() <= 1 {
        return (0..xs.len()).collect();
    }

    let mid = xs.len() / 2;
    let (left, right) = xs.split_at(mid);

    let (left_sorted, right_sorted) = rayon::join(
        || par_argsort(left),
        || par_argsort(right),
    );

    let left_candidates: Vec<Candidate> = left_sorted.iter().map(|&i| (xs[i], i)).collect();
    let right_candidates: Vec<Candidate> = right_sorted.iter().map(|&i| (xs[i + mid], i + mid)).collect();

    let merged_candidates = merge(&left_candidates, &right_candidates);

    merged_candidates.into_iter().map(|c| c.1).collect()
}

#[pyfunction]
fn par_argsort_module(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(par_argsort, m)?)?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn dotpro(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product_any, m)?)?;
    m.add_function(wrap_pyfunction!(par_argsort_module, m)?)?;
    Ok(())
}
