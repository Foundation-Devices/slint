// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Per-item caching of text layout state, shared by the text engines. `P` is
//! an engine's shaped representation of one item's text.

use crate::item_rendering::ItemCache;
use crate::items::TextWrap;
use crate::lengths::LogicalSize;

/// A measured text size, tagged with the `(max_width, wrap)` it was measured for.
#[derive(Clone)]
struct CachedTextSize {
    max_width: Option<f32>,
    wrap: TextWrap,
    size: Option<LogicalSize>,
}

/// Cache for shaped text and measured text sizes, keyed by ItemRc.
pub struct TextLayoutCache<P> {
    shaped: ItemCache<P>,
    size_cache: ItemCache<CachedTextSize>,
    #[cfg(feature = "testing")]
    cache_miss_count: core::cell::Cell<u64>,
}

#[allow(clippy::derivable_impls)] // clippy doesn't see the feature = "testing" code
impl<P> Default for TextLayoutCache<P> {
    fn default() -> Self {
        Self {
            shaped: Default::default(),
            size_cache: Default::default(),
            #[cfg(feature = "testing")]
            cache_miss_count: core::cell::Cell::new(0),
        }
    }
}

impl<P> TextLayoutCache<P> {
    /// The per-item shaped text; entries recompute when the item's properties
    /// change. Callers count a miss via [`Self::note_cache_miss`] in the
    /// update closure.
    pub fn shaped(&self) -> &ItemCache<P> {
        &self.shaped
    }

    pub fn note_cache_miss(&self) {
        #[cfg(feature = "testing")]
        self.cache_miss_count.set(self.cache_miss_count.get() + 1);
    }

    /// The cached text_size result for the item, measured by `measure` on a cache
    /// miss. The tracker recomputes the entry when the item's text/font/wrap
    /// change; the stored (max_width, wrap) guards against the query constraints
    /// differing between calls.
    pub fn cached_text_size(
        &self,
        item_rc: &crate::item_tree::ItemRc,
        max_width: Option<crate::lengths::LogicalLength>,
        wrap: TextWrap,
        measure: impl Fn() -> Option<LogicalSize>,
    ) -> Option<LogicalSize> {
        let max_width_key = max_width.map(|width| width.get());
        let cached = self.size_cache.get_or_update_cache_entry(item_rc, || CachedTextSize {
            max_width: max_width_key,
            wrap,
            size: measure(),
        });
        if cached.max_width == max_width_key && cached.wrap == wrap {
            cached.size
        } else {
            // Stale for these constraints, and the tracker can't see them, so drop it
            // and let the next call refill it.
            self.size_cache.release(item_rc);
            measure()
        }
    }

    pub fn clear_cache_if_scale_factor_changed(&self, window: &crate::api::Window) {
        self.shaped.clear_cache_if_scale_factor_changed(window);
        self.size_cache.clear_cache_if_scale_factor_changed(window);
    }

    pub fn component_destroyed(&self, component: crate::item_tree::ItemTreeRef) {
        self.shaped.component_destroyed(component);
        self.size_cache.component_destroyed(component);
    }

    pub fn clear_all(&self) {
        self.shaped.clear_all();
        self.size_cache.clear_all();
    }
}

#[cfg(feature = "testing")]
impl<P> TextLayoutCache<P> {
    pub fn cache_miss_count(&self) -> u64 {
        self.cache_miss_count.get()
    }
    pub fn reset_cache_miss_count(&self) {
        self.cache_miss_count.set(0);
    }
}
