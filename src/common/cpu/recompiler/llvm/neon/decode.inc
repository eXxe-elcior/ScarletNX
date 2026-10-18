/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
/* in snx_simd's order (cpu_helpers.c), each group's emitters in turn; 0 leaves the word to snx_simd */
int snx_ir_more_simd(SnxIr* ir, SnxU32 w) {
    if ((w & 0xBF800000UL) == 0x0C000000UL || (w & 0xBF800000UL) == 0x0C800000UL)
        return ld1_multiple(ir, w) || ld_structures(ir, w);
    if ((w & 0xBF000000UL) == 0x0D000000UL) return ld1_single(ir, w);
    if ((w & 0x9F200400UL) == 0x0E200400UL) return three_same(ir, w) || fp_three_same(ir, w, 0);
    if ((w & 0xDF200400UL) == 0x5E200400UL) return scalar_three_same(ir, w) || fp_three_same(ir, w, 1);
    if ((w & 0xBFE0FC00UL) == 0x0EE0E000UL) return pmull_1q(ir, w);
    if ((w & 0x9F200C00UL) == 0x0E200000UL) return three_different(ir, w);
    if ((w & 0xDF200C00UL) == 0x5E200000UL) return scalar_three_different(ir, w);
    if ((w & 0x9F3E0C00UL) == 0x0E200800UL)
        return byte_misc(ir, w) || fp_sign_misc(ir, w) || fp_misc(ir, w, 0) || fp_narrow(ir, w, 0) ||
               int_misc(ir, w, 0);
    if ((w & 0xDF3E0C00UL) == 0x5E200800UL) return fp_misc(ir, w, 1) || fp_narrow(ir, w, 1) || int_misc(ir, w, 1);
    if ((w & 0x9F3E0C00UL) == 0x0E300800UL) return across(ir, w);
    if ((w & 0xDF3E0C00UL) == 0x5E300800UL) return scalar_pairwise(ir, w);
    if ((w & 0x9FE08400UL) == 0x0E000400UL) return smov(ir, w);
    if ((w & 0xFFE0FC00UL) == 0x5E000400UL) return scalar_dup(ir, w);
    if ((w & 0x9FF80400UL) == 0x0F000400UL) return modified_immediate(ir, w);
    if ((w & 0x9F800400UL) == 0x0F000400UL) return fp_fixed(ir, w, 0) || shift_immediate(ir, w, 0);
    if ((w & 0xDF800400UL) == 0x5F000400UL) return fp_fixed(ir, w, 1) || shift_immediate(ir, w, 1);
    if ((w & 0x9F000400UL) == 0x0F000000UL) return by_element(ir, w, 0);
    if ((w & 0xDF000400UL) == 0x5F000000UL) return by_element(ir, w, 1);
    if ((w & 0xBF208C00UL) == 0x0E000800UL) return permute(ir, w);
    if ((w & 0xBFE08400UL) == 0x2E000000UL) return extract(ir, w);
    if ((w & 0xBFE08C00UL) == 0x0E000000UL) return table_lookup(ir, w);
    if ((w & 0xFFFF0C00UL) == 0x4E280800UL) return aes(ir, w);
    if ((w & 0xFFE08C00UL) == 0x5E000000UL) return sha(ir, w, (w >> 12) & 7);
    if ((w & 0xFFFF0C00UL) == 0x5E280800UL) return ((w >> 12) & 31) < 3 && sha(ir, w, 0x10 + ((w >> 12) & 31));
    return 0;
}
