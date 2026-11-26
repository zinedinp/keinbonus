//! An image as integer codes, plane by plane: what every layout packs from
//! and unpacks to, and what [`convert`](crate::convert) works on.

use alloc::vec;
use alloc::vec::Vec;

use crate::Error;
use crate::chroma::ChromaFormat;
use crate::quant::BitDepth;

/// Codes of one image, each plane row by row without padding.
///
/// `planes` are Y', Cb, Cr; for [`MatrixCoefficients::Identity`]
/// (RGB) they are G', B', R', H.273's order. Monochrome images have empty
/// colour planes. Alpha, when there is one, is full-resolution and full
/// range, at the image's bit depth.
///
/// [`MatrixCoefficients::Identity`]: crate::cicp::MatrixCoefficients::Identity
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub chroma: ChromaFormat,
    pub depth: BitDepth,
    pub planes: [Vec<u16>; 3],
    pub alpha: Option<Vec<u16>>,
}

impl Image {
    /// All codes 0, no alpha.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] for a width or height of 0.
    pub fn new(
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        depth: BitDepth,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::Size);
        }
        let (cw, ch) = chroma.plane(width, height);
        Ok(Self {
            width,
            height,
            chroma,
            depth,
            planes: [vec![0; width * height], vec![0; cw * ch], vec![0; cw * ch]],
            alpha: None,
        })
    }

    /// Width and height of colour plane 1 and 2.
    #[must_use]
    pub const fn chroma_size(&self) -> (usize, usize) {
        self.chroma.plane(self.width, self.height)
    }

    /// Plane sizes fit the format and every code fits the bit depth.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] or [`Error::Code`].
    pub fn check(&self) -> Result<(), Error> {
        let (cw, ch) = self.chroma_size();
        let luma = self.width * self.height;
        if self.width == 0
            || self.height == 0
            || self.planes[0].len() != luma
            || self.planes[1].len() != cw * ch
            || self.planes[2].len() != cw * ch
            || self.alpha.as_ref().is_some_and(|a| a.len() != luma)
        {
            return Err(Error::Size);
        }
        let max = self.depth.max();
        let over = self
            .planes
            .iter()
            .flatten()
            .chain(self.alpha.iter().flatten())
            .any(|&c| c > max);
        if over { Err(Error::Code) } else { Ok(()) }
    }
}
