#![cfg(target_arch = "arm")]
#![cfg(feature = "simd")]

use core::arch::arm::*;

// Implemented as a macro to ensure inlining and no function call overhead
macro_rules! div255_floor {
    ($x:ident) => {
        vshrn_n_u16::<7>(
            vuzpq_u16(
                vreinterpretq_u16_u32(vmull_n_u16(vget_low_u16($x), 0x8081)),
                vreinterpretq_u16_u32(vmull_n_u16(vget_high_u16($x), 0x8081))
            ).1
        )
    }
}

#[inline(always)]
pub(crate) unsafe fn simd_blend_8(fg_img: &[u32], bg_img: &[u32], dst_img: &mut [u32]) {
    // Load the four channels of 8 pixels of the foreground image into four
    // uint8x8 vector registers
    let fg = vld4_u8(fg_img.as_ptr().cast());

    // Same for the background image
    let mut bg = vld4_u8(bg_img.as_ptr().cast());

    // Byte order: Blue, Green, Red, Alpha
    let alpha   = fg.3;
    let alpha_c = vmvn_u8(alpha); // 255 - alpha

    // r = bg.r * (255 - alpha) + fg.r * alpha
    let r = vaddq_u16(vmull_u8(bg.2, alpha_c),
                             vmull_u8(fg.2, alpha));

    let g = vaddq_u16(vmull_u8(bg.1, alpha_c),
                             vmull_u8(fg.1, alpha));

    let b = vaddq_u16(vmull_u8(bg.0, alpha_c),
                             vmull_u8(fg.0, alpha));

    // Divide the 16-bit colors by 255 so the result is in [0, 255] again
    bg.2 = div255_floor!(r);
    bg.1 = div255_floor!(g);
    bg.0 = div255_floor!(b);

    // Store the four channels of 8 pixels to the output image
    vst4_u8(dst_img.as_mut_ptr().cast(), bg);
}

/// Byte shuffle lookup table for converting 24-bit (3 byte) into 32-bit (4 bytes) colors. Four at a time.
#[rustfmt::skip]
const LOOKUP_TABLE_24_TO_32: [u8; 16] = [
    0,  1,  2,  0xff,  3,  4,   5,   0xff,
    6,  7,  8,  0xff,  9,  10,  11,  0xff
];

#[inline(always)]
pub(crate) unsafe fn expand_24bpp_to_32bpp(
    alpha: u8,
    rgb24_pixels_packed: &[u8],
    rgb32_pixels: &mut [u32],
) {
    let input0 = vld1q_u8(rgb24_pixels_packed.as_ptr());
    let input1 = vld1q_u8(rgb24_pixels_packed.as_ptr().wrapping_add(12));
    let input2 = vld1q_u8(rgb24_pixels_packed.as_ptr().wrapping_add(24));
    let input3 = vld1q_u8(rgb24_pixels_packed.as_ptr().wrapping_add(36));

    let table = vld1q_u8(LOOKUP_TABLE_24_TO_32.as_ptr());

    let mut output0 = vdupq_n_u8(alpha);
    let mut output1 = vdupq_n_u8(alpha);
    let mut output2 = vdupq_n_u8(alpha);
    let mut output3 = vdupq_n_u8(alpha);
    core::arch::asm!(
        "vtbx.8 {output0:e}, {{ {input0:e}, {input0:f} }}, {table:e}",
        "vtbx.8 {output0:f}, {{ {input0:e}, {input0:f} }}, {table:f}",
        "vtbx.8 {output1:e}, {{ {input1:e}, {input1:f} }}, {table:e}",
        "vtbx.8 {output1:f}, {{ {input1:e}, {input1:f} }}, {table:f}",
        "vtbx.8 {output2:e}, {{ {input2:e}, {input2:f} }}, {table:e}",
        "vtbx.8 {output2:f}, {{ {input2:e}, {input2:f} }}, {table:f}",
        "vtbx.8 {output3:e}, {{ {input3:e}, {input3:f} }}, {table:e}",
        "vtbx.8 {output3:f}, {{ {input3:e}, {input3:f} }}, {table:f}",
        output0 = inout(qreg) output0,
        output1 = inout(qreg) output1,
        output2 = inout(qreg) output2,
        output3 = inout(qreg) output3,
        input0 = in(qreg) input0,
        input1 = in(qreg) input1,
        input2 = in(qreg) input2,
        input3 = in(qreg) input3,
        table = in(qreg) table,

        options(nostack)
    );

    vst1q_u8(rgb32_pixels.as_mut_ptr().cast(), output0);
    vst1q_u8(rgb32_pixels.as_mut_ptr().wrapping_add(4).cast(), output1);
    vst1q_u8(rgb32_pixels.as_mut_ptr().wrapping_add(8).cast(), output2);
    vst1q_u8(rgb32_pixels.as_mut_ptr().wrapping_add(12).cast(), output3);
}

/// Expands RGB to ARGB with predefined alpha, performs a 32-bit alpha-blending
/// then writes blended pixels back.
#[inline(always)]
pub(crate) unsafe fn expand_24bpp_to_32bpp_and_blend(
    alpha: u8,
    rgb24_pixels_packed: &[u8],
    rgb32_pixels: &mut [u32]
) {
    // let expanded = expand_24bpp_to_32bpp_inner(alpha, rgb24_pixels_packed);

    // TODO: alpha-blending
}

pub(crate) unsafe fn blend_by_alpha_map(alpha: u8, fg_color: u32, alpha_map: &[u8], bg_chunk: &mut [u32]) {
    // TODO: special case: multiply alpha (fg) with alpha_map and divide by 255 to get "combined" alpha

    let fg_b = vdup_n_u8((fg_color & 0xff) as u8);
    let fg_g = vdup_n_u8((fg_color >> 8 & 0xff) as u8);
    let fg_r = vdup_n_u8((fg_color >> 16 & 0xff) as u8);
    let fg_a = vld1_u8(alpha_map.as_ptr());
    let fg = uint8x8x4_t(fg_r, fg_g, fg_b, fg_a);

    // Same for the background image
    let mut bg = vld4_u8(bg_chunk.as_ptr().cast());

    // Byte order: Blue, Green, Red, Alpha
    let alpha   = fg.3;
    let alpha_c = vmvn_u8(alpha); // 255 - alpha

    // r = bg.r * (255 - alpha) + fg.r * alpha
    let r = vaddq_u16(vmull_u8(bg.2, alpha_c),
                      vmull_u8(fg.2, alpha));

    let g = vaddq_u16(vmull_u8(bg.1, alpha_c),
                      vmull_u8(fg.1, alpha));

    let b = vaddq_u16(vmull_u8(bg.0, alpha_c),
                      vmull_u8(fg.0, alpha));

    // Divide the 16-bit colors by 255 so the result is in [0, 255] again
    bg.2 = div255_floor!(r);
    bg.1 = div255_floor!(g);
    bg.0 = div255_floor!(b);

    // Store the four channels of 8 pixels to the output image
    vst4_u8(bg_chunk.as_mut_ptr().cast(), bg);
}
