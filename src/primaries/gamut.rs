//! Putting a source's colours on a display with other primaries: exactly
//! (each colour keeps its CIE xy) or expanded (the source's RGB taken as
//! the display's own, as a wide-gamut screen without colour management
//! shows sRGB).

use super::{Adaptation, Primaries};
use crate::math::Mat3;

/// How a source's colours meet a display's primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GamutMode {
    /// Colorimetric: every colour as the source means it.
    #[default]
    Exact,
    /// The source's primaries stretched to the display's: more saturated,
    /// hues near the primaries shifted.
    Expanded,
}

/// Linear RGB in `source` to linear RGB in `signal` (the colour space the
/// output is coded in, BT.2020 for HDR10), for a display whose own
/// primaries are `display`. [`GamutMode::Exact`] ignores `display`;
/// [`GamutMode::Expanded`] reads the source's RGB as the display's.
#[must_use]
pub fn source_to_signal(
    source: &Primaries,
    display: &Primaries,
    signal: &Primaries,
    mode: GamutMode,
) -> Mat3 {
    let from = match mode {
        GamutMode::Exact => source,
        GamutMode::Expanded => display,
    };
    from.conversion(signal, Adaptation::Bradford)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cicp::ColourPrimaries;
    use crate::math::{self, close};

    fn primaries(p: ColourPrimaries) -> Primaries {
        p.primaries().unwrap()
    }

    #[test]
    fn exact_keeps_white_and_expanded_reaches_the_displays_primaries() {
        let srgb = primaries(ColourPrimaries::Bt709);
        let p3 = primaries(ColourPrimaries::Smpte432);
        let bt2020 = primaries(ColourPrimaries::Bt2020);
        let exact = source_to_signal(&srgb, &p3, &bt2020, GamutMode::Exact);
        let white = math::apply(&exact, [1.0; 3]);
        assert!(white.iter().all(|&v| close(v, 1.0, 1e-9)), "{white:?}");
        // sRGB red inside BT.2020: green and blue stay positive.
        let red = math::apply(&exact, [1.0, 0.0, 0.0]);
        assert!(red.iter().all(|&v| v >= 0.0), "{red:?}");
        // Expanded: sRGB red is P3 red, coded in BT.2020.
        let expanded = source_to_signal(&srgb, &p3, &bt2020, GamutMode::Expanded);
        let p3_red = math::apply(&p3.conversion(&bt2020, Adaptation::Bradford), [1.0, 0.0, 0.0]);
        let red = math::apply(&expanded, [1.0, 0.0, 0.0]);
        assert!(red.iter().zip(p3_red).all(|(&a, b)| close(a, b, 1e-12)));
        // On an sRGB display both are the same.
        let same = source_to_signal(&srgb, &srgb, &bt2020, GamutMode::Expanded);
        assert!(same.iter().flatten().zip(exact.iter().flatten()).all(|(&a, &b)| close(a, b, 1e-12)));
    }
}
