//! The training environment as a Python module (`pokemon_ml._engine`),
//! built with `maturin develop --release`. Everything is handed over as
//! NumPy arrays that Python owns and Rust fills in place, with the
//! interpreter lock let go while the games are stepped.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::data::{ABILITIES, ITEMS, MOVES, SPECIES};
use crate::env::{Baseline, Config, N_ACTIONS, N_PREVIEW, OBS_M, VecEnv, preview_table};
use crate::obs::{self, OBS_F, OBS_I, Table};
use crate::teams::{Pool, Variation};

fn failed(e: crate::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:?}"))
}

fn whole<'a, T: numpy::Element>(a: &'a mut PyReadwriteArray2<'_, T>, what: &str) -> PyResult<&'a mut [T]> {
    a.as_slice_mut().map_err(|_| PyValueError::new_err(format!("{what} must be a C-contiguous array")))
}

/// Many games at once. See `pokemon_ml.env` for the Python side of it.
#[pyclass(name = "VecEnv")]
struct PyVecEnv {
    env: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    /// `pool` is the text of a team pool (`teams/<name>.json`, as `teampool` writes it).
    #[new]
    #[pyo3(signature = (pool, envs=256, seed=1, open_sheets=0.5, vary=true, max_turns=100, threads=0))]
    fn new(
        pool: &str,
        envs: usize,
        seed: u64,
        open_sheets: f64,
        vary: bool,
        max_turns: u16,
        threads: usize,
    ) -> PyResult<Self> {
        let pool = Pool::from_json(pool).map_err(failed)?;
        let variation = if vary { Variation::default() } else { Variation::NONE };
        let config = Config { envs, seed, open_sheets, variation, max_turns, threads };
        Ok(PyVecEnv { env: VecEnv::new(&pool, config).map_err(failed)? })
    }

    #[getter]
    fn envs(&self) -> usize {
        self.env.len()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.env.threads()
    }

    /// Fills the observations of every game as it stands: `f` (2·envs × OBS_F float32),
    /// `i` (2·envs × OBS_I int16), `mask` (2·envs × OBS_M uint8).
    fn observe(
        &self,
        py: Python<'_>,
        mut f: PyReadwriteArray2<'_, f32>,
        mut i: PyReadwriteArray2<'_, i16>,
        mut mask: PyReadwriteArray2<'_, u8>,
    ) -> PyResult<()> {
        let (f, i, mask) = (whole(&mut f, "f")?, whole(&mut i, "i")?, whole(&mut mask, "mask")?);
        self.sized(f.len(), i.len(), mask.len())?;
        let env = &self.env;
        py.detach(|| env.observe(f, i, mask));
        Ok(())
    }

    /// Both sides of every game act (`actions`: envs × 4 int32), and the observations, the
    /// results (`reward`: envs × 2 float32) and which games ended (`done`: envs uint8) are filled in.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray2<'_, i32>,
        mut f: PyReadwriteArray2<'_, f32>,
        mut i: PyReadwriteArray2<'_, i16>,
        mut mask: PyReadwriteArray2<'_, u8>,
        mut reward: PyReadwriteArray2<'_, f32>,
        mut done: PyReadwriteArray1<'_, u8>,
    ) -> PyResult<()> {
        let actions =
            actions.as_slice().map_err(|_| PyValueError::new_err("actions must be a C-contiguous int32 array"))?;
        let (f, i, mask) = (whole(&mut f, "f")?, whole(&mut i, "i")?, whole(&mut mask, "mask")?);
        let reward = whole(&mut reward, "reward")?;
        let done = done.as_slice_mut().map_err(|_| PyValueError::new_err("done must be a contiguous array"))?;
        self.sized(f.len(), i.len(), mask.len())?;
        let n = self.env.len();
        if actions.len() != 4 * n || reward.len() != 2 * n || done.len() != n {
            return Err(PyValueError::new_err(format!("actions must be {n} × 4, reward {n} × 2 and done {n}")));
        }
        let env = &mut self.env;
        py.detach(|| env.step(actions, f, i, mask, reward, done)).map_err(failed)
    }

    /// What a player that needs no model ("random" or "greedy") would do on each side of every game.
    fn baseline(&mut self, py: Python<'_>, kind: &str, mut actions: PyReadwriteArray2<'_, i32>) -> PyResult<()> {
        let kind = match kind {
            "random" => Baseline::Random,
            "greedy" => Baseline::Greedy,
            _ => return Err(PyValueError::new_err("the baselines are \"random\" and \"greedy\"")),
        };
        let actions = whole(&mut actions, "actions")?;
        if actions.len() != 4 * self.env.len() {
            return Err(PyValueError::new_err(format!("actions must be {} × 4", self.env.len())));
        }
        let env = &mut self.env;
        py.detach(|| env.baseline(kind, actions));
        Ok(())
    }

    /// Totals since the environment was made.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.env.stats();
        let d = PyDict::new(py);
        d.set_item("games", s.games)?;
        d.set_item("decisions", s.decisions)?;
        d.set_item("turns", s.turns)?;
        d.set_item("ties", s.ties)?;
        d.set_item("first_wins", s.first_wins)?;
        Ok(d)
    }
}

impl PyVecEnv {
    fn sized(&self, f: usize, i: usize, mask: usize) -> PyResult<()> {
        let n = 2 * self.env.len();
        if f != n * OBS_F || i != n * OBS_I || mask != n * OBS_M {
            return Err(PyValueError::new_err(format!(
                "observation arrays must be {n} × {OBS_F} float32, {n} × {OBS_I} int16 and {n} × {OBS_M} uint8"
            )));
        }
        Ok(())
    }
}

/// The sizes and the parts of an observation: for each part, where it begins and its shape.
#[pyfunction]
fn layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("obs_f", OBS_F)?;
    d.set_item("obs_i", OBS_I)?;
    d.set_item("obs_m", OBS_M)?;
    d.set_item("actions", N_ACTIONS)?;
    d.set_item("preview_actions", N_PREVIEW)?;
    d.set_item("roster", obs::ROSTER)?;
    let [species, items, abilities, moves] = obs::vocab();
    let vocab = PyDict::new(py);
    vocab.set_item("species", species)?;
    vocab.set_item("items", items)?;
    vocab.set_item("abilities", abilities)?;
    vocab.set_item("moves", moves)?;
    d.set_item("vocab", vocab)?;
    let (f, i) = obs::layout();
    for (key, parts) in [("f", f), ("i", i)] {
        let out = PyDict::new(py);
        for (name, at, shape) in parts {
            out.set_item(name, (at, shape))?;
        }
        d.set_item(key, out)?;
    }
    Ok(d)
}

fn array<'py>(py: Python<'py>, t: Table) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, t.data).reshape([t.rows, t.cols])?.into_any())
}

/// Static data for a model to embed ids with: a float32 table of features
/// for species, moves and items (row = id), and the four roster entries each
/// Team Preview action brings.
#[pyfunction]
fn tables(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("species", array(py, obs::species_table())?)?;
    d.set_item("moves", array(py, obs::move_table())?)?;
    d.set_item("items", array(py, obs::item_table())?)?;
    let preview: Vec<i64> = preview_table().iter().flatten().map(|&j| j as i64).collect();
    d.set_item("preview", PyArray1::from_vec(py, preview).reshape([N_PREVIEW, 4])?)?;
    Ok(d)
}

/// What each id is called (row = id; the first two are "nothing" and "unknown").
#[pyfunction]
fn names(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let list = |all: Vec<&'static str>| -> Vec<&'static str> { ["", "?"].into_iter().chain(all).collect() };
    let d = PyDict::new(py);
    d.set_item("species", list(SPECIES.iter().map(|s| s.name).collect()))?;
    d.set_item("items", list(ITEMS.iter().map(|s| s.name).collect()))?;
    d.set_item("abilities", list(ABILITIES.iter().map(|s| s.name).collect()))?;
    d.set_item("moves", list(MOVES.iter().map(|s| s.name).collect()))?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(layout, m)?)?;
    m.add_function(wrap_pyfunction!(tables, m)?)?;
    m.add_function(wrap_pyfunction!(names, m)?)?;
    Ok(())
}
