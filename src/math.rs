//! The arithmetic every reference function shares: H.273's `Round` and
//! `Clip3`, 3x3 matrices, and the transcendental functions through `libm`
//! (the same bits on every platform).

/// A 3x3 matrix, rows times a column vector.
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// H.273's `Round(x) = Sign(x) * Floor(Abs(x) + 0.5)`: halves away from 0.
#[must_use]
pub fn round(x: f64) -> f64 {
    libm::floor(x.abs() + 0.5).copysign(x)
}

/// H.273's `Clip3(lo, hi, x)`.
#[must_use]
pub fn clip3(lo: f64, hi: f64, x: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[must_use]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// `pow` extended to negative `x` as an odd function: `-pow(-x, y)`.
#[must_use]
pub fn pow_odd(x: f64, y: f64) -> f64 {
    libm::pow(x.abs(), y).copysign(x)
}

#[must_use]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[must_use]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[must_use]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

#[must_use]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[must_use]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[must_use]
pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

#[must_use]
pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The inverse, or `None` for a singular matrix. `const`, so fixed
/// matrices get their inverse at compile time.
#[must_use]
pub const fn invert(m: &Mat3) -> Option<Mat3> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let cof = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let det = a * cof[0][0] + b * cof[1][0] + c * cof[2][0];
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    let mut r = 0;
    while r < 3 {
        let mut k = 0;
        while k < 3 {
            out[r][k] = cof[r][k] / det;
            k += 1;
        }
        r += 1;
    }
    Some(out)
}

/// [`invert`] of a matrix known to be invertible, for constants.
///
/// # Panics
///
/// When `m` is singular: at compile time where it initializes a constant.
#[must_use]
pub const fn inverse_of(m: &Mat3) -> Mat3 {
    match invert(m) {
        Some(inv) => inv,
        None => panic!("singular matrix"),
    }
}

#[must_use]
pub fn diag(v: [f64; 3]) -> Mat3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

#[cfg(test)]
pub(crate) fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

#[cfg(test)]
pub(crate) fn mat_close(a: &Mat3, b: &Mat3, eps: f64) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(&x, &y)| close(x, y, eps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_takes_halves_away_from_zero() {
        assert_eq!(round(0.5), 1.0);
        assert_eq!(round(1.5), 2.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(-0.5), -1.0);
        assert_eq!(round(0.499_999_999), 0.0);
        assert_eq!(clip3(0.0, 255.0, 256.0), 255.0);
        assert_eq!(clip3(0.0, 255.0, -1.0), 0.0);
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let m = [[2.0, 1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]];
        let inv = invert(&m).unwrap();
        assert!(mat_close(&mul(&inv, &m), &IDENTITY, 1e-15));
        assert!(invert(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]).is_none());
        assert!(close(pow_odd(-8.0, 1.0 / 3.0), -2.0, 1e-15));
    }
}
