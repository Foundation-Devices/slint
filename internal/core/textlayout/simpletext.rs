// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! The simple text engine: glyphs come from the character map plus the mini
//! shaper's ligatures and kerning, lines from the classic line breaker. It
//! mirrors the function surface of `sharedparley`, drives renderers through
//! the same [`GlyphRenderer`] trait, and caches shaped text per item.

pub mod minishaper;
pub use minishaper::MiniShaper;

use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::pin::Pin;

use i_slint_common::sharedfontique::fontique;
use skrifa::MetadataProvider as _;

use super::glyphrenderer::{GlyphRenderer, PhysicalLength, PhysicalRect, RenderGlyph};
use super::{
    DEFAULT_FONT_SIZE, FontMetrics, Glyph, ShapeBuffer, TextLayout, TextParagraphLayout, TextShaper,
};
use crate::SharedString;
use crate::graphics::FontRequest;
use crate::item_rendering::{HasFont, PlainOrStyledText};
use crate::items::{TextOverflow, TextWrap};
use crate::lengths::{
    LogicalBorderRadius, LogicalLength, LogicalPoint, LogicalRect, LogicalSize, PointLengths,
    ScaleFactor, SizeLengths,
};
use crate::renderer::RendererSealed;
use euclid::num::Zero;

type PhysicalSize = euclid::Size2D<f32, crate::lengths::PhysicalPx>;
type PhysicalPoint = euclid::Point2D<f32, crate::lengths::PhysicalPx>;

/// A font face at a fixed pixel size, shaping through the mini shaper.
pub struct SimpleFont {
    blob: fontique::Blob<u8>,
    index: u32,
    pixel_size: PhysicalLength,
    units_per_em: f32,
    ascent: PhysicalLength,
    descent: PhysicalLength,
    x_height: PhysicalLength,
    cap_height: PhysicalLength,
}

impl SimpleFont {
    pub fn new(blob: fontique::Blob<u8>, index: u32, pixel_size: PhysicalLength) -> Option<Self> {
        let face = skrifa::FontRef::from_index(blob.data(), index).ok()?;
        let metrics = face
            .metrics(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::default());
        let units_per_em = metrics.units_per_em as f32;
        if units_per_em <= 0.0 {
            return None;
        }
        let scale = pixel_size.get() / units_per_em;
        Some(Self {
            blob,
            index,
            pixel_size,
            units_per_em,
            ascent: PhysicalLength::new(metrics.ascent * scale),
            descent: PhysicalLength::new(metrics.descent * scale),
            x_height: PhysicalLength::new(metrics.x_height.unwrap_or_default() * scale),
            cap_height: PhysicalLength::new(metrics.cap_height.unwrap_or_default() * scale),
        })
    }

    fn face(&self) -> skrifa::FontRef<'_> {
        // The construction already parsed this data, so it cannot fail here.
        skrifa::FontRef::from_index(self.blob.data(), self.index).unwrap()
    }

    fn matched_for_request(
        font_request: &FontRequest,
        scale_factor: ScaleFactor,
        font_context: &mut super::FontContext,
    ) -> Option<Self> {
        let pixel_size = font_request.pixel_size.unwrap_or(DEFAULT_FONT_SIZE) * scale_factor;
        let font = font_request.query_fontique(
            &mut font_context.inner.collection,
            &mut font_context.inner.source_cache,
        )?;
        Self::new(font.blob, font.index, pixel_size)
    }
}

impl TextShaper for SimpleFont {
    type LengthPrimitive = f32;
    type Length = PhysicalLength;

    fn shape_text<GlyphStorage: core::iter::Extend<Glyph<PhysicalLength>>>(
        &self,
        text: &str,
        glyphs: &mut GlyphStorage,
    ) {
        let face = self.face();
        let charmap = face.charmap();
        let glyph_metrics = skrifa::metrics::GlyphMetrics::new(
            &face,
            skrifa::instance::Size::new(self.pixel_size.get()),
            skrifa::instance::LocationRef::default(),
        );
        let shaper = MiniShaper::new(&face);
        let kern_scale = self.pixel_size.get() / self.units_per_em;

        // Buffered because ligature substitution splices the stream.
        let mut shaped: Vec<Glyph<PhysicalLength>> = text
            .char_indices()
            .map(|(byte_offset, ch)| Glyph {
                glyph_id: charmap.map(ch).and_then(|id| NonZeroU16::new(id.to_u32() as u16)),
                text_byte_offset: byte_offset,
                ..Default::default()
            })
            .collect();

        shaper.substitute_ligatures(&mut shaped);

        // Advances only after substitution; a ligature glyph has its own width.
        for glyph in &mut shaped {
            glyph.advance = PhysicalLength::new(match glyph.glyph_id {
                Some(id) => glyph_metrics
                    .advance_width(skrifa::GlyphId::new(id.get() as u32))
                    .unwrap_or_default(),
                None => self.pixel_size.get(),
            });
        }

        for i in 1..shaped.len() {
            if let (Some(left), Some(right)) = (shaped[i - 1].glyph_id, shaped[i].glyph_id) {
                let kern = shaper.kern(left, right) as f32 * kern_scale;
                shaped[i - 1].advance += PhysicalLength::new(kern);
            }
        }

        glyphs.extend(shaped);
    }

    fn glyph_for_char(&self, ch: char) -> Option<Glyph<PhysicalLength>> {
        let face = self.face();
        let glyph_id = face.charmap().map(ch).and_then(|id| NonZeroU16::new(id.to_u32() as u16))?;
        let glyph_metrics = skrifa::metrics::GlyphMetrics::new(
            &face,
            skrifa::instance::Size::new(self.pixel_size.get()),
            skrifa::instance::LocationRef::default(),
        );
        Some(Glyph {
            glyph_id: Some(glyph_id),
            advance: PhysicalLength::new(
                glyph_metrics
                    .advance_width(skrifa::GlyphId::new(glyph_id.get() as u32))
                    .unwrap_or_default(),
            ),
            ..Default::default()
        })
    }

    fn max_lines(&self, max_height: PhysicalLength) -> usize {
        (max_height.get() / self.height().get()) as usize
    }
}

impl FontMetrics<PhysicalLength> for SimpleFont {
    fn ascent(&self) -> PhysicalLength {
        self.ascent
    }

    fn descent(&self) -> PhysicalLength {
        self.descent
    }

    fn x_height(&self) -> PhysicalLength {
        self.x_height
    }

    fn cap_height(&self) -> PhysicalLength {
        self.cap_height
    }
}

/// A text item's string shaped with the font matched for it. `None` when no
/// font matched the request.
pub type MaybeShape = Option<Rc<CachedShape>>;

pub struct CachedShape {
    /// The shaped text; the buffer's glyph offsets index into it.
    string: String,
    font: SimpleFont,
    buffer: ShapeBuffer<PhysicalLength>,
}

/// Cache for shaped text, keyed by ItemRc.
pub type TextLayoutCache = super::layoutcache::TextLayoutCache<MaybeShape>;

fn physical_letter_spacing(
    font_request: &FontRequest,
    scale_factor: ScaleFactor,
) -> Option<PhysicalLength> {
    font_request.letter_spacing.map(|spacing| spacing * scale_factor)
}

/// Shapes the item's text with the font matched for it. Property reads happen
/// up front so a caching caller's update closure registers the dependencies.
fn shape_text_item(
    slint_ctx: &crate::SlintContext,
    text: Pin<&dyn crate::item_rendering::RenderText>,
    item_rc: Option<&crate::item_tree::ItemRc>,
    scale_factor: ScaleFactor,
) -> MaybeShape {
    let font_request = item_rc.map(|item_rc| text.font_request(item_rc)).unwrap_or_default();
    let string = match &text.text() {
        PlainOrStyledText::Plain(string) => string.to_string(),
        PlainOrStyledText::Styled(styled_text) => {
            crate::styled_text::get_raw_text(styled_text).into_owned()
        }
    };

    let mut font_ctx = slint_ctx.font_context().borrow_mut();
    let font = SimpleFont::matched_for_request(&font_request, scale_factor, &mut font_ctx)?;
    drop(font_ctx);

    let buffer = ShapeBuffer::new(
        &TextLayout {
            font: &font,
            letter_spacing: physical_letter_spacing(&font_request, scale_factor),
        },
        &string,
    );

    Some(Rc::new(CachedShape { string, font, buffer }))
}

fn get_or_shape(
    cache: Option<&TextLayoutCache>,
    slint_ctx: &crate::SlintContext,
    text: Pin<&dyn crate::item_rendering::RenderText>,
    item_rc: Option<&crate::item_tree::ItemRc>,
    scale_factor: ScaleFactor,
) -> MaybeShape {
    if let (Some(cache), Some(item_rc)) = (cache, item_rc) {
        cache.shaped().get_or_update_cache_entry(item_rc, || {
            cache.note_cache_miss();
            shape_text_item(slint_ctx, text, Some(item_rc), scale_factor)
        })
    } else {
        shape_text_item(slint_ctx, text, item_rc, scale_factor)
    }
}

/// Emits one line of glyphs through the renderer, splitting around the
/// selection so selected glyphs draw in the selection's foreground brush.
#[allow(clippy::too_many_arguments)]
fn draw_line_glyphs<R: GlyphRenderer>(
    item_renderer: &mut R,
    font: &SimpleFont,
    glyphs: &mut dyn Iterator<Item = super::PositionedGlyph<PhysicalLength>>,
    line_x: PhysicalLength,
    baseline_y: PhysicalLength,
    fill_brush: &R::PlatformBrush,
    selection: Option<(&core::ops::Range<usize>, &R::PlatformBrush)>,
) {
    let mut run: Vec<RenderGlyph> = Vec::new();
    let mut run_selected = false;
    let flush = |item_renderer: &mut R, run: &mut Vec<RenderGlyph>, selected: bool| {
        if run.is_empty() {
            return;
        }
        let brush = if selected { selection.unwrap().1 } else { fill_brush };
        item_renderer.draw_glyph_run(
            &font.blob,
            font.index,
            font.pixel_size,
            &[],
            &fontique::Synthesis::default(),
            brush.clone(),
            PhysicalLength::default(),
            &mut run.drain(..),
        );
    };

    for glyph in glyphs {
        let selected = selection.is_some_and(|(range, _)| range.contains(&glyph.text_byte_offset));
        if selected != run_selected {
            flush(item_renderer, &mut run, run_selected);
            run_selected = selected;
        }
        run.push(RenderGlyph {
            id: glyph.glyph_id.get() as u32,
            x: (line_x + glyph.x).get(),
            y: baseline_y.get(),
            advance: glyph.advance.get(),
        });
    }
    flush(item_renderer, &mut run, run_selected);
}

pub fn draw_text(
    item_renderer: &mut impl GlyphRenderer,
    text: Pin<&dyn crate::item_rendering::RenderText>,
    item_rc: Option<&crate::item_tree::ItemRc>,
    size: LogicalSize,
    cache: Option<&TextLayoutCache>,
) {
    if size.width <= 0. || size.height <= 0. {
        return;
    }

    let Some(fill_brush) = item_renderer.platform_text_fill_brush(text.color(), size) else {
        return;
    };

    let scale_factor = ScaleFactor::new(item_renderer.scale_factor());
    let slint_ctx = item_renderer.window().context().clone();

    let Some(shape) = get_or_shape(cache, &slint_ctx, text, item_rc, scale_factor) else {
        return;
    };
    if shape.string.trim().is_empty() {
        return;
    }

    let font_request = item_rc.map(|item_rc| text.font_request(item_rc)).unwrap_or_default();
    let (horizontal_alignment, vertical_alignment) = text.alignment();
    let max_size: PhysicalSize = size * scale_factor;

    let paragraph = TextParagraphLayout {
        string: &shape.string,
        layout: TextLayout {
            font: &shape.font,
            letter_spacing: physical_letter_spacing(&font_request, scale_factor),
        },
        max_width: max_size.width_length(),
        max_height: max_size.height_length(),
        horizontal_alignment,
        vertical_alignment,
        wrap: text.wrap(),
        overflow: text.overflow(),
        single_line: false,
    };

    let clip = text.overflow() == TextOverflow::Clip;
    let render = if clip {
        item_renderer.save_state();
        item_renderer.combine_clip(
            LogicalRect::new(LogicalPoint::default(), size),
            LogicalBorderRadius::zero(),
            LogicalLength::zero(),
        )
    } else {
        true
    };

    if render {
        paragraph
            .layout_lines::<()>(
                &shape.buffer,
                |glyphs, line_x, line_y, _, _| {
                    let baseline_y = line_y + shape.font.ascent();
                    draw_line_glyphs(
                        item_renderer,
                        &shape.font,
                        glyphs,
                        line_x,
                        baseline_y,
                        &fill_brush,
                        None,
                    );
                    core::ops::ControlFlow::Continue(())
                },
                None,
            )
            .ok();
    }

    if clip {
        item_renderer.restore_state();
    }
}

pub fn draw_text_input(
    item_renderer: &mut impl GlyphRenderer,
    text_input: Pin<&crate::items::TextInput>,
    item_rc: &crate::item_tree::ItemRc,
    size: LogicalSize,
    _cache: Option<&TextLayoutCache>,
) {
    if size.width <= 0. || size.height <= 0. {
        return;
    }

    let scale_factor = ScaleFactor::new(item_renderer.scale_factor());
    let slint_ctx = item_renderer.window().context().clone();

    let font_request = text_input.font_request(item_rc);
    let mut font_ctx = slint_ctx.font_context().borrow_mut();
    let Some(font) = SimpleFont::matched_for_request(&font_request, scale_factor, &mut font_ctx)
    else {
        return;
    };
    drop(font_ctx);

    let visual_representation = text_input.visual_representation(None);

    let Some(fill_brush) =
        item_renderer.platform_text_fill_brush(visual_representation.text_color.clone(), size)
    else {
        return;
    };
    let selection_brush = (!visual_representation.selection_range.is_empty())
        .then(|| item_renderer.platform_brush_for_color(&text_input.selection_foreground_color()))
        .flatten();

    let max_size: PhysicalSize = size * scale_factor;

    let paragraph = TextParagraphLayout {
        string: &visual_representation.text,
        layout: TextLayout {
            font: &font,
            letter_spacing: physical_letter_spacing(&font_request, scale_factor),
        },
        max_width: max_size.width_length(),
        max_height: max_size.height_length(),
        horizontal_alignment: text_input.horizontal_alignment(),
        vertical_alignment: text_input.vertical_alignment(),
        wrap: text_input.wrap(),
        overflow: TextOverflow::Clip,
        single_line: text_input.single_line(),
    };

    item_renderer.save_state();
    let render = item_renderer.combine_clip(
        LogicalRect::new(LogicalPoint::default(), size),
        LogicalBorderRadius::zero(),
        LogicalLength::zero(),
    );

    if render {
        let selection_background = text_input.selection_background_color();
        let selection =
            selection_brush.as_ref().map(|brush| (&visual_representation.selection_range, brush));

        paragraph
            .layout_lines::<()>(
                &paragraph.shape(),
                |glyphs, line_x, line_y, _, line_selection| {
                    if let Some(sel) = line_selection {
                        item_renderer.fill_rectangle_with_color(
                            PhysicalRect::new(
                                PhysicalPoint::from_lengths(line_x + sel.start, line_y),
                                PhysicalSize::from_lengths(sel.end - sel.start, font.height()),
                            ),
                            selection_background,
                        );
                    }
                    let baseline_y = line_y + font.ascent();
                    draw_line_glyphs(
                        item_renderer,
                        &font,
                        glyphs,
                        line_x,
                        baseline_y,
                        &fill_brush,
                        selection,
                    );
                    core::ops::ControlFlow::Continue(())
                },
                Some(visual_representation.selection_range.clone()),
            )
            .ok();

        if let Some(cursor_offset) = visual_representation.cursor_position {
            let (cursor_x, cursor_y) = paragraph.cursor_pos_for_byte_offset(cursor_offset);
            item_renderer.fill_rectangle_with_color(
                PhysicalRect::new(
                    PhysicalPoint::from_lengths(cursor_x, cursor_y),
                    PhysicalSize::from_lengths(
                        text_input.text_cursor_width() * scale_factor,
                        font.height(),
                    ),
                ),
                visual_representation.cursor_color,
            );
        }
    }

    item_renderer.restore_state();
}

pub fn measure_text_size(
    renderer: &dyn RendererSealed,
    text_item: Pin<&dyn crate::item_rendering::RenderString>,
    item_rc: &crate::item_tree::ItemRc,
    max_width: Option<LogicalLength>,
    text_wrap: TextWrap,
) -> Option<LogicalSize> {
    let scale_factor = renderer.scale_factor()?;
    let ctx = renderer.slint_context()?;

    // Read the text/font properties before borrowing font_context: both can trigger bindings
    // that re-enter text_size for other elements and panic on a second borrow_mut(). Called
    // from the size cache's update closure, these reads also register the cache dependencies.
    let font_request = text_item.font_request(item_rc);
    let content = text_item.text();
    let string = match &content {
        PlainOrStyledText::Plain(string) => alloc::borrow::Cow::Borrowed(string.as_str()),
        PlainOrStyledText::Styled(styled_text) => crate::styled_text::get_raw_text(styled_text),
    };

    let mut font_ctx = ctx.font_context().borrow_mut();
    let font = SimpleFont::matched_for_request(&font_request, scale_factor, &mut font_ctx)?;
    drop(font_ctx);

    let layout = TextLayout {
        font: &font,
        letter_spacing: physical_letter_spacing(&font_request, scale_factor),
    };
    let (longest_line_width, height) =
        layout.text_size(&string, max_width.map(|width| width * scale_factor), text_wrap);

    Some(PhysicalSize::from_lengths(longest_line_width, height) / scale_factor)
}

fn text_input_font(
    renderer: &dyn RendererSealed,
    text_input: Pin<&crate::items::TextInput>,
    item_rc: &crate::item_tree::ItemRc,
    scale_factor: ScaleFactor,
) -> Option<(SimpleFont, FontRequest)> {
    let ctx = renderer.slint_context()?;
    let font_request = text_input.font_request(item_rc);
    let mut font_ctx = ctx.font_context().borrow_mut();
    let font = SimpleFont::matched_for_request(&font_request, scale_factor, &mut font_ctx)?;
    Some((font, font_request))
}

fn text_input_paragraph<'a>(
    font: &'a SimpleFont,
    font_request: &FontRequest,
    text_input: Pin<&crate::items::TextInput>,
    string: &'a str,
    scale_factor: ScaleFactor,
) -> TextParagraphLayout<'a, SimpleFont> {
    let max_size: PhysicalSize =
        LogicalSize::from_lengths(text_input.width(), text_input.height()) * scale_factor;
    TextParagraphLayout {
        string,
        layout: TextLayout {
            font,
            letter_spacing: physical_letter_spacing(font_request, scale_factor),
        },
        max_width: max_size.width_length(),
        max_height: max_size.height_length(),
        horizontal_alignment: text_input.horizontal_alignment(),
        vertical_alignment: text_input.vertical_alignment(),
        wrap: text_input.wrap(),
        overflow: TextOverflow::Clip,
        single_line: false,
    }
}

/// The byte offset into `visual_text` for a click at `pos`, in physical pixels.
/// `None` when the renderer has no context or no font matches.
pub fn visual_text_byte_offset_for_position(
    renderer: &dyn RendererSealed,
    text_input: Pin<&crate::items::TextInput>,
    item_rc: &crate::item_tree::ItemRc,
    visual_text: &SharedString,
    pos: PhysicalPoint,
    scale_factor: ScaleFactor,
) -> Option<usize> {
    let (font, font_request) = text_input_font(renderer, text_input, item_rc, scale_factor)?;
    let paragraph =
        text_input_paragraph(&font, &font_request, text_input, visual_text, scale_factor);
    Some(paragraph.byte_offset_for_position((pos.x_length(), pos.y_length())))
}

/// The cursor rectangle in physical pixels for a byte offset into `visual_text`.
/// `None` when the renderer has no context or no font matches.
pub fn visual_text_cursor_rect_for_byte_offset(
    renderer: &dyn RendererSealed,
    text_input: Pin<&crate::items::TextInput>,
    item_rc: &crate::item_tree::ItemRc,
    visual_text: &SharedString,
    byte_offset: usize,
    cursor_width: PhysicalLength,
    scale_factor: ScaleFactor,
) -> Option<PhysicalRect> {
    let (font, font_request) = text_input_font(renderer, text_input, item_rc, scale_factor)?;
    let paragraph =
        text_input_paragraph(&font, &font_request, text_input, visual_text, scale_factor);
    let (cursor_x, cursor_y) = paragraph.cursor_pos_for_byte_offset(byte_offset);
    Some(PhysicalRect::new(
        PhysicalPoint::from_lengths(cursor_x, cursor_y),
        PhysicalSize::from_lengths(cursor_width, font.height()),
    ))
}
