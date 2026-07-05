// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! The interface between the text engines and the renderers: an engine shapes
//! and positions glyphs, a renderer rasterizes and blits them.

use i_slint_common::sharedfontique::fontique;

use crate::Color;
use crate::lengths::{LogicalSize, PhysicalPx};

pub type PhysicalLength = euclid::Length<f32, PhysicalPx>;
pub type PhysicalRect = euclid::Rect<f32, PhysicalPx>;

/// A glyph positioned by a text engine, ready for rasterization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderGlyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub advance: f32,
}

/// Trait used for drawing text and text input elements, where a text engine does
/// the shaping and positioning, and the renderer is responsible for drawing just
/// the glyphs.
pub trait GlyphRenderer: crate::item_rendering::ItemRenderer {
    /// A renderer-specific type for a brush used for fill and stroke of glyphs.
    type PlatformBrush: Clone;

    /// Returns the brush to be used for filling text.
    fn platform_text_fill_brush(
        &mut self,
        brush: crate::Brush,
        size: LogicalSize,
    ) -> Option<Self::PlatformBrush>;

    /// Returns a brush that's a solid fill of the specified color.
    fn platform_brush_for_color(&mut self, color: &Color) -> Option<Self::PlatformBrush>;

    /// Returns the brush to be used for stroking text.
    fn platform_text_stroke_brush(
        &mut self,
        brush: crate::Brush,
        physical_stroke_width: f32,
        size: LogicalSize,
    ) -> Option<Self::PlatformBrush>;

    /// Draws the glyphs provided by glyphs_it from the font face at `font_index`
    /// inside `font_blob`, with the specified font_size and brush at the given y
    /// offset. The `normalized_coords` are F2Dot14 values in fvar axis order for
    /// variable font rendering. The `synthesis` contains design-space variation
    /// settings and faux bold/italic hints from fontique.
    fn draw_glyph_run(
        &mut self,
        font_blob: &fontique::Blob<u8>,
        font_index: u32,
        font_size: PhysicalLength,
        normalized_coords: &[i16],
        synthesis: &fontique::Synthesis,
        brush: Self::PlatformBrush,
        y_offset: PhysicalLength,
        glyphs_it: &mut dyn Iterator<Item = RenderGlyph>,
    );

    fn fill_rectangle_with_color(&mut self, physical_rect: PhysicalRect, color: Color) {
        if let Some(platform_brush) = self.platform_brush_for_color(&color) {
            self.fill_rectangle(physical_rect, platform_brush);
        }
    }

    /// Fills the given rectangle with the specified color. This is used for drawing selection
    /// rectangles as well as the text cursor.
    fn fill_rectangle(&mut self, physical_rect: PhysicalRect, brush: Self::PlatformBrush);
}
