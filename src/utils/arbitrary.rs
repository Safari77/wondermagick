//! Helper functions for using `quickcheck`'s `Arbitrary` trait

use quickcheck::Arbitrary;

#[must_use]
pub fn optional_positive_float(g: &mut quickcheck::Gen) -> Option<f64> {
    if bool::arbitrary(g) {
        Some(positive_float(g))
    } else {
        None
    }
}

#[must_use]
pub fn positive_float(g: &mut quickcheck::Gen) -> f64 {
    nonzero_float(g).abs()
}

#[must_use]
pub fn finite_float(g: &mut quickcheck::Gen) -> f64 {
    let raw = f64::arbitrary(g);
    if raw.is_infinite() || raw.is_nan() {
        0.0
    } else {
        raw
    }
}

#[must_use]
pub fn nonzero_float(g: &mut quickcheck::Gen) -> f64 {
    let float = finite_float(g);
    if float == 0.0 || float == -0.0 {
        1.0
    } else {
        float
    }
}

#[must_use]
pub fn optional_nonzero_float(g: &mut quickcheck::Gen) -> Option<f64> {
    if bool::arbitrary(g) {
        Some(nonzero_float(g))
    } else {
        None
    }
}
