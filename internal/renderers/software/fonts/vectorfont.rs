// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

use core::num::NonZeroU16;

use ab_glyph::Font;
use alloc::rc::Rc;
use skrifa::MetadataProvider;

use crate::PhysicalLength;
use crate::fixed::Fixed;
use i_slint_common::sharedfontique::fontique;
use i_slint_core::lengths::PhysicalPx;
use i_slint_core::textlayout::{Glyph, TextShaper};

use super::RenderableVectorGlyph;

// A length in font design space.
struct FontUnit;
type FontLength = euclid::Length<i32, FontUnit>;
type FontScaleFactor = euclid::Scale<f32, FontUnit, PhysicalPx>;

type GlyphCacheKey = (u64, u32, PhysicalLength, core::num::NonZeroU16);

struct RenderableGlyphWeightScale;

impl clru::WeightScale<GlyphCacheKey, RenderableVectorGlyph> for RenderableGlyphWeightScale {
    fn weight(&self, _: &GlyphCacheKey, value: &RenderableVectorGlyph) -> usize {
        value.alpha_map.len()
    }
}

type GlyphCache = clru::CLruCache<
    GlyphCacheKey,
    RenderableVectorGlyph,
    std::collections::hash_map::RandomState,
    RenderableGlyphWeightScale,
>;

i_slint_core::thread_local!(static GLYPH_CACHE: core::cell::RefCell<GlyphCache>  =
    core::cell::RefCell::new(
        clru::CLruCache::with_config(
            clru::CLruCacheConfig::new(core::num::NonZeroUsize::new(1024 * 1024).unwrap())
                .with_scale(RenderableGlyphWeightScale)
        )
    )
);

pub struct VectorFont {
    font_index: u32,
    font_blob: fontique::Blob<u8>,
    ascender: PhysicalLength,
    descender: PhysicalLength,
    height: PhysicalLength,
    pixel_size: PhysicalLength,
    x_height: PhysicalLength,
    cap_height: PhysicalLength,
}

impl VectorFont {
    pub fn new(font: fontique::QueryFont, pixel_size: PhysicalLength) -> Self {
        Self::new_from_blob_and_index(font.blob, font.index, pixel_size)
    }

    pub fn new_from_blob_and_index(
        font_blob: fontique::Blob<u8>,
        font_index: u32,
        pixel_size: PhysicalLength,
    ) -> Self {
        let face = skrifa::FontRef::from_index(font_blob.data(), font_index).unwrap();

        let metrics = face
            .metrics(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::new(&[]));

        let ascender = FontLength::new(metrics.ascent as _);
        let descender = FontLength::new(metrics.descent as _);
        let height = FontLength::new((metrics.ascent - metrics.descent) as _);
        let x_height = FontLength::new(metrics.x_height.unwrap_or_default() as _);
        let cap_height = FontLength::new(metrics.cap_height.unwrap_or_default() as _);
        let units_per_em = metrics.units_per_em;
        let scale = FontScaleFactor::new(pixel_size.get() as f32 / units_per_em as f32);
        Self {
            font_index,
            font_blob,
            ascender: (ascender.cast() * scale).cast(),
            descender: (descender.cast() * scale).cast(),
            height: (height.cast() * scale).cast(),
            pixel_size,
            x_height: (x_height.cast() * scale).cast(),
            cap_height: (cap_height.cast() * scale).cast(),
        }
    }

    pub fn render_vector_glyph(
        &self,
        glyph_id: core::num::NonZeroU16,
    ) -> Option<RenderableVectorGlyph> {
        GLYPH_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            let cache_key = (self.font_blob.id(), self.font_index, self.pixel_size, glyph_id);

            if let Some(entry) = cache.get(&cache_key) {
                Some(entry.clone())
            } else {
                // Note: Creating a new ab_glyph object for every glyph rendering can
                //       seem wasteful (and it is), but due to lifetimes, we can't
                //       create cached FontRefs, and owning Fonts waste a lot of memory.
                //       Fortunately parsing is relatively cheap, so we can actually
                //       afford to do this, especially since we cache the rendered
                //       glyphs themselves.
                let face = ab_glyph::FontRef::try_from_slice_and_index(
                    self.font_blob.as_ref(),
                    self.font_index,
                )
                .ok()?;
                let outline = face.outline_glyph(ab_glyph::Glyph {
                    id: ab_glyph::GlyphId(glyph_id.get()),
                    // ab_glyph uses a weird "font height" metric, so we need to transform
                    // pixel sizes to that here.
                    scale: (face.height_unscaled() / face.units_per_em()?
                        * (self.pixel_size.get() as f32))
                        .into(),
                    position: Default::default(),
                })?;

                let bounds = outline.px_bounds();
                let mut alpha_map = alloc::vec![0u8; (bounds.width() * bounds.height()) as usize];
                outline.draw(|x, y, value| {
                    alpha_map[y as usize * bounds.width() as usize + x as usize] =
                        (value * 255.0) as u8;
                });
                let alpha_map: Rc<[u8]> = alpha_map.into();

                let glyph = super::RenderableVectorGlyph {
                    x: Fixed::from_f32(bounds.min.x)?,
                    y: Fixed::from_f32(-bounds.max.y)?,
                    width: PhysicalLength::new(bounds.width() as i16),
                    height: PhysicalLength::new(bounds.height() as i16),
                    alpha_map,
                    pixel_stride: bounds.width() as u16,
                };

                cache.put_with_weight(cache_key, glyph.clone()).ok();
                Some(glyph)
            }
        })
    }
}

impl TextShaper for VectorFont {
    type LengthPrimitive = i16;
    type Length = PhysicalLength;
    fn shape_text<GlyphStorage: core::iter::Extend<Glyph<PhysicalLength>>>(
        &self,
        text: &str,
        glyphs: &mut GlyphStorage,
    ) {
        let Ok(face) =
            ab_glyph::FontRef::try_from_slice_and_index(self.font_blob.as_ref(), self.font_index)
        else {
            return;
        };
        glyphs.extend(text.char_indices().map(|(byte_offset, char)| {
            let raw_glyph_id = face.glyph_id(char);
            let glyph_id = NonZeroU16::try_from(raw_glyph_id.0).ok();
            let x_advance = glyph_id.map_or_else(
                || self.pixel_size.get(),
                |_id| {
                    (face.h_advance_unscaled(raw_glyph_id) / face.units_per_em().unwrap_or(1.0)
                        * (self.pixel_size.get() as f32)) as _
                },
            );

            Glyph {
                glyph_id,
                advance: PhysicalLength::new(x_advance),
                text_byte_offset: byte_offset,
                ..Default::default()
            }
        }));
    }

    fn glyph_for_char(&self, ch: char) -> Option<Glyph<PhysicalLength>> {
        let face =
            ab_glyph::FontRef::try_from_slice_and_index(self.font_blob.as_ref(), self.font_index)
                .ok()?;
        let raw_glyph_id = face.glyph_id(ch);
        NonZeroU16::try_from(raw_glyph_id.0).ok().map(|glyph_id| {
            let mut out_glyph = Glyph::default();
            out_glyph.glyph_id = Some(glyph_id);
            out_glyph.advance = PhysicalLength::new(
                (face.h_advance_unscaled(raw_glyph_id) / face.units_per_em().unwrap_or(1.0)
                    * (self.pixel_size.get() as f32)) as _,
            );
            out_glyph
        })
    }

    fn max_lines(&self, max_height: PhysicalLength) -> usize {
        (max_height / self.height).get() as _
    }
}

impl i_slint_core::textlayout::FontMetrics<PhysicalLength> for VectorFont {
    fn ascent(&self) -> PhysicalLength {
        self.ascender
    }

    fn height(&self) -> PhysicalLength {
        self.height
    }

    fn descent(&self) -> PhysicalLength {
        self.descender
    }

    fn x_height(&self) -> PhysicalLength {
        self.x_height
    }

    fn cap_height(&self) -> PhysicalLength {
        self.cap_height
    }
}

impl super::GlyphRenderer for VectorFont {
    fn render_glyph(&self, glyph_id: core::num::NonZeroU16) -> Option<super::RenderableGlyph> {
        self.render_vector_glyph(glyph_id).map(|glyph| super::RenderableGlyph {
            x: glyph.x,
            y: glyph.y,
            width: glyph.width,
            height: glyph.height,
            alpha_map: glyph.alpha_map.into(),
            pixel_stride: glyph.pixel_stride,
            sdf: false,
        })
    }

    fn scale_delta(&self) -> super::Fixed<u16, 8> {
        super::Fixed::from_integer(1)
    }
}
