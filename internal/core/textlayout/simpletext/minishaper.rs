// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Kerning and ligatures for the simple (non-parley) text path.
//!
//! A full shaper costs hundreds of kilobytes for scripts we never render.
//! This module applies just the two OpenType lookups Latin UI fonts actually
//! use: ligature substitution (GSUB type 4, feature `liga`) and pair
//! positioning (GPOS type 2, feature `kern`), unwrapping the Extension
//! lookups (GSUB type 7, GPOS type 9) that fonts with large tables wrap
//! them in. Other lookup types and malformed tables are skipped, so an
//! exotic font degrades to unkerned text, never a panic.

use alloc::vec::Vec;
use core::num::NonZeroU16;

use crate::textlayout::Glyph;
use skrifa::raw::TableProvider;
use skrifa::raw::tables::gpos::{PairPos, PositionSubtables};
use skrifa::raw::tables::gsub::{LigatureSubstFormat1, SubstitutionSubtables};
use skrifa::raw::tables::layout::FeatureList;
use skrifa::raw::types::{GlyphId16, Tag};

pub struct MiniShaper<'a> {
    /// One entry per `liga` lookup, in font order; an entry is the lookup's subtables.
    liga: Vec<Vec<LigatureSubstFormat1<'a>>>,
    /// Same, for `kern` lookups.
    kern: Vec<Vec<PairPos<'a>>>,
}

impl<'a> MiniShaper<'a> {
    pub fn new(font: &skrifa::FontRef<'a>) -> Self {
        let mut shaper = Self { liga: Vec::new(), kern: Vec::new() };
        if let Ok(gsub) = font.gsub() {
            if let (Ok(features), Ok(lookups)) = (gsub.feature_list(), gsub.lookup_list()) {
                for index in feature_lookup_indices(&features, Tag::new(b"liga")) {
                    if let Ok(SubstitutionSubtables::Ligature(subtables)) =
                        lookups.lookups().get(index as usize).and_then(|lookup| lookup.subtables())
                    {
                        shaper.liga.push(subtables.iter().filter_map(Result::ok).collect());
                    }
                }
            }
        }
        if let Ok(gpos) = font.gpos() {
            if let (Ok(features), Ok(lookups)) = (gpos.feature_list(), gpos.lookup_list()) {
                for index in feature_lookup_indices(&features, Tag::new(b"kern")) {
                    if let Ok(PositionSubtables::Pair(subtables)) =
                        lookups.lookups().get(index as usize).and_then(|lookup| lookup.subtables())
                    {
                        shaper.kern.push(subtables.iter().filter_map(Result::ok).collect());
                    }
                }
            }
        }
        shaper
    }

    /// Replaces ligature component sequences (fi, fl, ...) in place. A merged
    /// glyph keeps the byte offset of its first character. Advances must be
    /// assigned afterwards, because the ligature glyph has its own width.
    pub fn substitute_ligatures<Length>(&self, glyphs: &mut Vec<Glyph<Length>>) {
        for lookup in &self.liga {
            let mut i = 0;
            while i < glyphs.len() {
                if let Some((ligature, component_count)) = ligature_at(lookup, glyphs, i) {
                    glyphs[i].glyph_id = NonZeroU16::new(ligature.to_u16());
                    glyphs.drain(i + 1..i + component_count);
                }
                i += 1;
            }
        }
    }

    /// The `kern` adjustment to the advance of `left` when followed by
    /// `right`, in font units.
    pub fn kern(&self, left: NonZeroU16, right: NonZeroU16) -> i32 {
        let (left, right) = (GlyphId16::new(left.get()), GlyphId16::new(right.get()));
        // Lookups accumulate; within one, the first subtable covering the pair wins.
        self.kern
            .iter()
            .filter_map(|lookup| lookup.iter().find_map(|s| pair_x_advance(s, left, right)))
            .map(i32::from)
            .sum()
    }
}

/// Lookup indices registered for `tag`, across every script in the font.
/// Proper resolution walks script -> language system -> feature, but UI fonts
/// register the same Latin lookups everywhere, so the walk would only ever
/// find duplicates.
fn feature_lookup_indices(features: &FeatureList, tag: Tag) -> Vec<u16> {
    let mut indices = Vec::new();
    for record in features.feature_records() {
        if record.feature_tag() != tag {
            continue;
        }
        let Ok(feature) = record.feature(features.offset_data()) else { continue };
        for index in feature.lookup_list_indices() {
            // Scripts share lookups; applying a kern lookup twice would double it.
            if !indices.contains(&index.get()) {
                indices.push(index.get());
            }
        }
    }
    indices
}

/// The ligature starting at `glyphs[at]`, as (ligature glyph, component count).
fn ligature_at<Length>(
    lookup: &[LigatureSubstFormat1],
    glyphs: &[Glyph<Length>],
    at: usize,
) -> Option<(GlyphId16, usize)> {
    let first = gid(glyphs.get(at)?)?;
    for subtable in lookup {
        let Ok(coverage) = subtable.coverage() else { continue };
        let Some(set_index) = coverage.get(first) else { continue };
        let Ok(set) = subtable.ligature_sets().get(set_index as usize) else { continue };
        // Ligatures within a set are ordered by preference; take the first full match.
        for ligature in set.ligatures().iter().filter_map(Result::ok) {
            let components = ligature.component_glyph_ids();
            let matches = glyphs.get(at + 1..at + 1 + components.len()).is_some_and(|rest| {
                components
                    .iter()
                    .zip(rest)
                    .all(|(component, glyph)| gid(glyph) == Some(component.get()))
            });
            if matches {
                return Some((ligature.ligature_glyph(), components.len() + 1));
            }
        }
    }
    None
}

/// The pair adjustment for (left, right) from one PairPos subtable, or `None`
/// when the subtable does not cover the pair. Only the first glyph's x
/// advance is read; fonts put horizontal kerning there (valueFormat1 =
/// X_ADVANCE, valueFormat2 = 0).
fn pair_x_advance(subtable: &PairPos, left: GlyphId16, right: GlyphId16) -> Option<i16> {
    match subtable {
        PairPos::Format1(table) => {
            let set_index = table.coverage().ok()?.get(left)?;
            let set = table.pair_sets().get(set_index as usize).ok()?;
            set.pair_value_records()
                .iter()
                .filter_map(Result::ok)
                .find(|record| record.second_glyph() == right)
                .map(|record| record.value_record1().x_advance().unwrap_or(0))
        }
        PairPos::Format2(table) => {
            // A coverage hit always applies in format 2; unlisted glyphs land
            // in class 0, whose value is usually zero.
            table.coverage().ok()?.get(left)?;
            let class1 = table.class_def1().ok()?.get(left);
            let class2 = table.class_def2().ok()?.get(right);
            let record = table.class1_records().get(class1 as usize).ok()?;
            let record = record.class2_records().get(class2 as usize).ok()?;
            Some(record.value_record1().x_advance().unwrap_or(0))
        }
    }
}

fn gid<Length>(glyph: &Glyph<Length>) -> Option<GlyphId16> {
    glyph.glyph_id.map(|id| GlyphId16::new(id.get()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NotoSans wraps its kern lookup in a GPOS Extension lookup, so it also
    /// covers the unwrapping path.
    fn noto_sans() -> std::vec::Vec<u8> {
        let path: std::path::PathBuf = [
            env!("CARGO_MANIFEST_DIR"),
            "..",
            "..",
            "tests",
            "screenshots",
            "fonts",
            "NotoSans-Regular.ttf",
        ]
        .iter()
        .collect();
        std::fs::read(path).unwrap()
    }

    fn shape<'a>(font: &skrifa::FontRef<'a>, text: &str) -> Vec<Glyph<i16>> {
        use skrifa::MetadataProvider;
        text.char_indices()
            .map(|(text_byte_offset, ch)| Glyph {
                glyph_id: font.charmap().map(ch).and_then(|id| NonZeroU16::new(id.to_u32() as u16)),
                text_byte_offset,
                ..Default::default()
            })
            .collect()
    }

    fn reference_shape(face: &rustybuzz::Face, text: &str) -> (Vec<u32>, Vec<i32>) {
        let mut buffer = rustybuzz::UnicodeBuffer::new();
        buffer.push_str(text);
        let shaped = rustybuzz::shape(face, &[], buffer);
        (
            shaped.glyph_infos().iter().map(|info| info.glyph_id).collect(),
            shaped.glyph_positions().iter().map(|position| position.x_advance).collect(),
        )
    }

    #[test]
    fn kerns_and_ligates_like_rustybuzz() {
        let data = noto_sans();
        let font = skrifa::FontRef::new(&data).unwrap();
        let shaper = MiniShaper::new(&font);
        let face = rustybuzz::Face::from_slice(&data, 0).unwrap();

        // The kern pair adjustment must equal the shift rustybuzz applies to
        // the first glyph's advance.
        let mut kerned_pairs = 0;
        for text in ["AV", "To", "Av", "Ta", "Yo", "nn", "ll"] {
            let glyphs = shape(&font, text);
            let (ids, advances) = reference_shape(&face, text);
            let raw_advance = face
                .glyph_hor_advance(rustybuzz::ttf_parser::GlyphId(ids[0] as u16))
                .unwrap() as i32;
            let reference_kern = advances[0] - raw_advance;
            kerned_pairs += (reference_kern != 0) as u32;
            assert_eq!(
                shaper.kern(glyphs[0].glyph_id.unwrap(), glyphs[1].glyph_id.unwrap()),
                reference_kern,
                "{text}"
            );
        }
        assert!(kerned_pairs > 0, "the fixture font kerns none of the pairs");

        // The unmappable char must not confuse the ligature matcher.
        let mut ligated_texts = 0;
        for text in ["fill", "fi", "fl", "ffi", "f\u{20BF}i"] {
            let mut glyphs = shape(&font, text);
            shaper.substitute_ligatures(&mut glyphs);
            let ids: Vec<u32> =
                glyphs.iter().map(|glyph| glyph.glyph_id.map_or(0, |id| id.get() as u32)).collect();
            let (reference_ids, _) = reference_shape(&face, text);
            ligated_texts += (reference_ids.len() < text.chars().count()) as u32;
            assert_eq!(ids, reference_ids, "{text}");
        }
        assert!(ligated_texts > 0, "the fixture font ligates none of the texts");

        // A merged glyph keeps the first component's byte offset; glyphs
        // after it keep theirs.
        let mut glyphs = shape(&font, "fill");
        shaper.substitute_ligatures(&mut glyphs);
        assert_eq!(glyphs[0].text_byte_offset, 0);
        assert_eq!(glyphs[1].text_byte_offset, 2);
    }
}
