// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Font database state and metrics queries that only need fontique and
//! skrifa, shared between the parley text engine and the simple one.

use std::collections::HashSet;
use std::sync::Arc;

use core::pin::Pin;

use i_slint_common::sharedfontique;
use i_slint_common::sharedfontique::fontique;
#[cfg(feature = "shared-parley")]
use parley;
use skrifa::MetadataProvider as _;

use crate::graphics::FontRequest;
use crate::lengths::{LogicalLength, LogicalSize};

pub use super::DEFAULT_FONT_SIZE;

/// The font database that font matching and the text engines share: the
/// fontique collection and its source cache. The two field layouts have the
/// same surface, so `inner.collection` and `inner.source_cache` read the same
/// in every configuration; with parley enabled, `inner` is the exact type
/// parley's entry points take.
#[cfg(not(feature = "shared-parley"))]
pub struct FontStore {
    pub collection: fontique::Collection,
    pub source_cache: fontique::SourceCache,
}

/// The state behind `SlintContext::font_context`.
pub struct FontContext {
    #[cfg(feature = "shared-parley")]
    pub inner: parley::FontContext,
    #[cfg(not(feature = "shared-parley"))]
    pub inner: FontStore,
    /// `(ptr, len)` of each `&'static [u8]` already handed to fontique, so repeat
    /// `register_static_font` calls for the same embedded font are skipped.
    registered_static_fonts: HashSet<(usize, usize)>,
}

impl FontContext {
    pub fn new(collection: sharedfontique::Collection) -> Self {
        #[cfg(feature = "shared-parley")]
        let inner = parley::FontContext {
            collection: collection.inner,
            source_cache: collection.source_cache,
        };
        #[cfg(not(feature = "shared-parley"))]
        let inner =
            FontStore { collection: collection.inner, source_cache: collection.source_cache };
        Self { inner, registered_static_fonts: HashSet::default() }
    }

    pub fn register_static_font(&mut self, data: &'static [u8]) {
        let key = (data.as_ptr() as usize, data.len());
        if self.registered_static_fonts.insert(key) {
            self.inner.collection.register_fonts(fontique::Blob::new(Arc::new(data)), None);
        }
    }

    pub fn clear_registered_static_fonts(&mut self) {
        self.registered_static_fonts.clear();
    }

    pub fn set_default_font_family(&mut self, family_name: &str) -> bool {
        sharedfontique::set_default_font_family(&mut self.inner.collection, family_name)
    }
}

/// The size of `ch` in the font matched for the item: advance width by line height.
pub fn char_size(
    collection: &mut fontique::Collection,
    source_cache: &mut fontique::SourceCache,
    text_item: Pin<&dyn crate::item_rendering::HasFont>,
    item_rc: &crate::item_tree::ItemRc,
    ch: char,
) -> Option<LogicalSize> {
    let font_request = text_item.font_request(item_rc);
    let font = font_request.query_fontique(collection, source_cache)?;

    let char_map = font.charmap()?;

    let face = skrifa::FontRef::from_index(font.blob.data(), font.index).unwrap();

    let glyph_index = char_map.map(ch)?;

    let pixel_size = font_request.pixel_size.unwrap_or(DEFAULT_FONT_SIZE);

    let location = face.axes().location(font.synthesis.variation_settings());

    let glyph_metrics = skrifa::metrics::GlyphMetrics::new(
        &face,
        skrifa::instance::Size::new(pixel_size.get()),
        &location,
    );

    let advance_width = LogicalLength::new(glyph_metrics.advance_width(glyph_index.into())?);

    let font_metrics = skrifa::metrics::Metrics::new(
        &face,
        skrifa::instance::Size::new(pixel_size.get()),
        &location,
    );

    Some(LogicalSize::from_lengths(
        advance_width,
        LogicalLength::new(font_metrics.ascent - font_metrics.descent),
    ))
}

/// Metrics of the font matched for `font_request`, scaled to its pixel size.
pub fn font_metrics(
    collection: &mut fontique::Collection,
    source_cache: &mut fontique::SourceCache,
    font_request: FontRequest,
) -> crate::items::FontMetrics {
    let logical_pixel_size = font_request.pixel_size.unwrap_or(DEFAULT_FONT_SIZE).get();

    let Some(font) = font_request.query_fontique(collection, source_cache) else {
        return crate::items::FontMetrics::default();
    };

    let face = skrifa::FontRef::from_index(font.blob.data(), font.index).unwrap();
    let location = face.axes().location(font.synthesis.variation_settings());
    let metrics = face.metrics(skrifa::instance::Size::unscaled(), &location);

    let units_per_em = metrics.units_per_em as f32;

    crate::items::FontMetrics {
        ascent: metrics.ascent * logical_pixel_size / units_per_em,
        descent: metrics.descent * logical_pixel_size / units_per_em,
        x_height: metrics.x_height.unwrap_or_default() * logical_pixel_size / units_per_em,
        cap_height: metrics.cap_height.unwrap_or_default() * logical_pixel_size / units_per_em,
    }
}
