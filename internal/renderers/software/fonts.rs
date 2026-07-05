// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

// cSpell: ignore pixelfont vectorfont
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

use super::{Fixed, PhysicalLength, PhysicalPoint, PhysicalRect, PhysicalSize};
use i_slint_core::graphics::{BitmapFont, FontRequest};
use i_slint_core::items::{TextOverflow, TextWrap};
use i_slint_core::lengths::{
    LogicalLength, LogicalPoint, LogicalRect, LogicalSize, PointLengths, ScaleFactor,
};
use i_slint_core::textlayout::{TextLayout, TextParagraphLayout};

i_slint_core::thread_local! {
    static BITMAP_FONTS: RefCell<Vec<&'static BitmapFont>> = RefCell::default()
}

#[derive(derive_more::From, Clone)]
pub enum GlyphAlphaMap {
    Static(&'static [u8]),
    Shared(Rc<[u8]>),
}

#[derive(Clone)]
pub struct RenderableGlyph {
    pub x: Fixed<i32, 8>,
    pub y: Fixed<i32, 8>,
    pub width: PhysicalLength,
    pub height: PhysicalLength,
    pub alpha_map: GlyphAlphaMap,
    pub pixel_stride: u16,
    pub sdf: bool,
}

impl RenderableGlyph {
    pub fn size(&self) -> PhysicalSize {
        PhysicalSize::from_lengths(self.width, self.height)
    }
}

// Subset of `RenderableGlyph`, specifically for VectorFonts.
#[cfg(feature = "systemfonts")]
#[derive(Clone)]
pub struct RenderableVectorGlyph {
    pub x: Fixed<i32, 8>,
    pub y: Fixed<i32, 8>,
    pub width: PhysicalLength,
    pub height: PhysicalLength,
    pub alpha_map: Rc<[u8]>,
    pub pixel_stride: u16,
    pub glyph_origin_x: f32,
}

#[cfg(feature = "systemfonts")]
impl RenderableVectorGlyph {
    pub fn size(&self) -> PhysicalSize {
        PhysicalSize::from_lengths(self.width, self.height)
    }
}

pub trait GlyphRenderer {
    fn render_glyph(
        &self,
        glyph_id: core::num::NonZeroU16,
        slint_context: &i_slint_core::SlintContext,
    ) -> Option<RenderableGlyph>;
    /// The amount of pixel in the original image that correspond to one pixel in the rendered image
    fn scale_delta(&self) -> Fixed<u16, 8>;
}

pub(super) use i_slint_core::textlayout::DEFAULT_FONT_SIZE;

mod pixelfont;
#[cfg(feature = "systemfonts")]
pub mod vectorfont;

#[cfg(feature = "systemfonts")]
pub mod systemfonts;

#[derive(derive_more::From)]
pub enum Font {
    PixelFont(pixelfont::PixelFont),
    #[cfg(feature = "systemfonts")]
    VectorFont(vectorfont::VectorFont),
}

/// Returns the size of the pre-rendered font in pixels.
pub fn pixel_size(glyphs: &i_slint_core::graphics::BitmapGlyphs) -> PhysicalLength {
    PhysicalLength::new(glyphs.pixel_size)
}

impl i_slint_core::textlayout::FontMetrics<PhysicalLength> for Font {
    fn ascent(&self) -> PhysicalLength {
        match self {
            Font::PixelFont(pixel_font) => pixel_font.ascent(),
            #[cfg(feature = "systemfonts")]
            Font::VectorFont(vector_font) => vector_font.ascent(),
        }
    }

    fn height(&self) -> PhysicalLength {
        match self {
            Font::PixelFont(pixel_font) => pixel_font.height(),
            #[cfg(feature = "systemfonts")]
            Font::VectorFont(vector_font) => vector_font.height(),
        }
    }

    fn descent(&self) -> PhysicalLength {
        match self {
            Font::PixelFont(pixel_font) => pixel_font.descent(),
            #[cfg(feature = "systemfonts")]
            Font::VectorFont(vector_font) => vector_font.descent(),
        }
    }

    fn x_height(&self) -> PhysicalLength {
        match self {
            Font::PixelFont(pixel_font) => pixel_font.x_height(),
            #[cfg(feature = "systemfonts")]
            Font::VectorFont(vector_font) => vector_font.x_height(),
        }
    }

    fn cap_height(&self) -> PhysicalLength {
        match self {
            Font::PixelFont(pixel_font) => pixel_font.cap_height(),
            #[cfg(feature = "systemfonts")]
            Font::VectorFont(vector_font) => vector_font.cap_height(),
        }
    }
}

pub fn match_font(
    request: &FontRequest,
    scale_factor: ScaleFactor,
    #[cfg(feature = "systemfonts")] font_context: &mut i_slint_core::textlayout::FontContext,
) -> Font {
    let requested_weight = request
        .weight
        .and_then(|weight| weight.try_into().ok())
        .unwrap_or(/* CSS normal */ 400);

    let bitmap_font = BITMAP_FONTS.with(|fonts| {
        let fonts = fonts.borrow();

        request.family.as_ref().and_then(|requested_family| {
            fonts
                .iter()
                .filter(|bitmap_font| {
                    core::str::from_utf8(bitmap_font.family_name.as_slice()).unwrap()
                        == requested_family.as_str()
                        && bitmap_font.italic == request.italic
                })
                .min_by_key(|bitmap_font| bitmap_font.weight.abs_diff(requested_weight))
                .copied()
        })
    });

    let font = match bitmap_font {
        Some(bitmap_font) => bitmap_font,
        None => {
            #[cfg(feature = "systemfonts")]
            if let Some(vectorfont) = systemfonts::match_font(
                request,
                scale_factor,
                &mut font_context.inner.collection,
                &mut font_context.inner.source_cache,
            ) {
                return vectorfont.into();
            }
            if let Some(fallback_bitmap_font) = BITMAP_FONTS.with(|fonts| {
                let fonts = fonts.borrow();
                fonts
                    .iter()
                    .cloned()
                    .filter(|bitmap_font| bitmap_font.italic == request.italic)
                    .min_by_key(|bitmap_font| bitmap_font.weight.abs_diff(requested_weight))
                    .or_else(|| fonts.first().cloned())
            }) {
                fallback_bitmap_font
            } else {
                #[cfg(feature = "systemfonts")]
                return systemfonts::fallbackfont(
                    request,
                    scale_factor,
                    &mut font_context.inner.collection,
                    &mut font_context.inner.source_cache,
                )
                .into();
                #[cfg(not(feature = "systemfonts"))]
                panic!(
                    "No font fallback found. The software renderer requires enabling the `EmbedForSoftwareRenderer` option when compiling slint files."
                )
            }
        }
    };

    let requested_pixel_size: PhysicalLength =
        (request.pixel_size.unwrap_or(DEFAULT_FONT_SIZE).cast() * scale_factor).cast();

    let nearest_pixel_size = font
        .glyphs
        .partition_point(|glyphs| pixel_size(glyphs) <= requested_pixel_size)
        .saturating_sub(1);
    let matching_glyphs = &font.glyphs[nearest_pixel_size];

    let pixel_size = if font.sdf { requested_pixel_size } else { pixel_size(matching_glyphs) };

    pixelfont::PixelFont { bitmap_font: font, glyphs: matching_glyphs, pixel_size }.into()
}

pub fn text_layout_for_font<'a, Font>(
    font: &'a Font,
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
) -> TextLayout<'a, Font>
where
    Font: i_slint_core::textlayout::AbstractFont
        + i_slint_core::textlayout::TextShaper<Length = PhysicalLength>,
{
    let letter_spacing =
        font_request.letter_spacing.map(|spacing| (spacing.cast() * scale_factor).cast());

    TextLayout { font, letter_spacing }
}

pub fn register_bitmap_font(font_data: &'static BitmapFont) {
    BITMAP_FONTS.with(|fonts| fonts.borrow_mut().push(font_data))
}

/// The layout size of `string` in logical pixels, wrapped at `max_width`.
pub fn text_size_with_font<Font>(
    font: &Font,
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
    string: &str,
    max_width: Option<LogicalLength>,
    text_wrap: TextWrap,
) -> LogicalSize
where
    Font: i_slint_core::textlayout::AbstractFont
        + i_slint_core::textlayout::TextShaper<Length = PhysicalLength>,
{
    let layout = text_layout_for_font(font, font_request, scale_factor);
    let (longest_line_width, height) = layout.text_size(
        string,
        max_width.map(|max_width| (max_width.cast() * scale_factor).cast()),
        text_wrap,
    );
    (PhysicalSize::from_lengths(longest_line_width, height).cast() / scale_factor).cast()
}

/// The layout size of a single character in logical pixels.
pub fn char_size_with_font<Font>(
    font: &Font,
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
    ch: char,
) -> LogicalSize
where
    Font: i_slint_core::textlayout::AbstractFont
        + i_slint_core::textlayout::TextShaper<Length = PhysicalLength>,
{
    let mut buf = [0u8; 4];
    text_size_with_font(
        font,
        font_request,
        scale_factor,
        ch.encode_utf8(&mut buf),
        None,
        TextWrap::NoWrap,
    )
}

/// The font's metrics converted to logical pixels.
pub fn font_metrics_with_font(
    font: &impl i_slint_core::textlayout::FontMetrics<PhysicalLength>,
    scale_factor: ScaleFactor,
) -> i_slint_core::items::FontMetrics {
    let ascent: LogicalLength = (font.ascent().cast() / scale_factor).cast();
    let descent: LogicalLength = (font.descent().cast() / scale_factor).cast();
    let x_height: LogicalLength = (font.x_height().cast() / scale_factor).cast();
    let cap_height: LogicalLength = (font.cap_height().cast() / scale_factor).cast();

    i_slint_core::items::FontMetrics {
        ascent: ascent.get() as _,
        descent: descent.get() as _,
        x_height: x_height.get() as _,
        cap_height: cap_height.get() as _,
    }
}

/// The byte offset in the text input's actual text for a click at `pos`.
pub fn text_input_byte_offset_for_position_with_font<Font>(
    font: &Font,
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
    text_input: core::pin::Pin<&i_slint_core::items::TextInput>,
    pos: LogicalPoint,
) -> usize
where
    Font: i_slint_core::textlayout::AbstractFont
        + i_slint_core::textlayout::TextShaper<Length = PhysicalLength>,
{
    let visual_representation = text_input.visual_representation(None);

    let width = (text_input.width().cast() * scale_factor).cast();
    let height = (text_input.height().cast() * scale_factor).cast();

    let pos = (pos.cast() * scale_factor)
        .clamp(euclid::point2(0., 0.), euclid::point2(i16::MAX, i16::MAX).cast())
        .cast();

    let layout = text_layout_for_font(font, font_request, scale_factor);

    let paragraph = TextParagraphLayout {
        string: &visual_representation.text,
        layout,
        max_width: width,
        max_height: height,
        horizontal_alignment: text_input.horizontal_alignment(),
        vertical_alignment: text_input.vertical_alignment(),
        wrap: text_input.wrap(),
        overflow: TextOverflow::Clip,
        single_line: false,
    };

    visual_representation.map_byte_offset_from_visual_text_to_actual_text(
        paragraph.byte_offset_for_position((pos.x_length(), pos.y_length())),
    )
}

/// The cursor rectangle for a byte offset into the text input, in logical pixels.
pub fn text_input_cursor_rect_for_byte_offset_with_font<Font>(
    font: &Font,
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
    text_input: core::pin::Pin<&i_slint_core::items::TextInput>,
    byte_offset: usize,
) -> LogicalRect
where
    Font: i_slint_core::textlayout::AbstractFont
        + i_slint_core::textlayout::TextShaper<Length = PhysicalLength>,
{
    let visual_representation = text_input.visual_representation(None);

    let width = (text_input.width().cast() * scale_factor).cast();
    let height = (text_input.height().cast() * scale_factor).cast();

    let layout = text_layout_for_font(font, font_request, scale_factor);

    let paragraph = TextParagraphLayout {
        string: &visual_representation.text,
        layout,
        max_width: width,
        max_height: height,
        horizontal_alignment: text_input.horizontal_alignment(),
        vertical_alignment: text_input.vertical_alignment(),
        wrap: text_input.wrap(),
        overflow: TextOverflow::Clip,
        single_line: false,
    };

    let cursor_position = paragraph.cursor_pos_for_byte_offset(byte_offset);
    let cursor_height = font.height();

    (PhysicalRect::new(
        PhysicalPoint::from_lengths(cursor_position.0, cursor_position.1),
        PhysicalSize::from_lengths(
            (text_input.text_cursor_width().cast() * scale_factor).cast(),
            cursor_height,
        ),
    )
    .cast()
        / scale_factor)
        .cast()
}
