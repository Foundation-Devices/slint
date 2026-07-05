// SPDX-FileCopyrightText: 2026 Foundation Devices, Inc. <hello@foundation.xyz>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Per-item caching of text layout state, shared by the text engines. `P` is
//! an engine's shaped representation of one item's text.

use crate::item_rendering::ItemCache;

/// Cache for shaped text and measured text sizes, keyed by ItemRc.
pub struct TextLayoutCache<P> {
    shaped: ItemCache<P>,
    #[cfg(feature = "testing")]
    cache_miss_count: core::cell::Cell<u64>,
}

#[allow(clippy::derivable_impls)] // clippy doesn't see the feature = "testing" code
impl<P> Default for TextLayoutCache<P> {
    fn default() -> Self {
        Self {
            shaped: Default::default(),
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

    pub fn clear_cache_if_scale_factor_changed(&self, window: &crate::api::Window) {
        self.shaped.clear_cache_if_scale_factor_changed(window);
    }

    pub fn component_destroyed(&self, component: crate::item_tree::ItemTreeRef) {
        self.shaped.component_destroyed(component);
    }

    pub fn clear_all(&self) {
        self.shaped.clear_all();
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
