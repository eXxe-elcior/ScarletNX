/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
static int host_ternlog(SnxIr* ir) { return ir->features->x86 && ir->features->avx512vl; }
/* VPTERNLOG{D,Q} on 128 bits: lanes of 32 or 64 bits (only the masking granularity differs) */
static V ternlog(SnxIr* ir, V a, V b, V c, unsigned table, unsigned bits) {
    V args[4];
    LLVMTypeRef t = LLVMVectorType(bits == 64 ? ir->t_i64 : ir->t_i32, bits == 64 ? 2 : 4);
    args[0] = LLVMBuildBitCast(B, a, t, "");
    args[1] = LLVMBuildBitCast(B, b, t, "");
    args[2] = LLVMBuildBitCast(B, c, t, "");
    args[3] = ir_i32(ir, table & 0xFF);
    return ir_intrinsic(ir, bits == 64 ? "llvm.x86.avx512.pternlog.q.128" : "llvm.x86.avx512.pternlog.d.128", NULL,
                        args, 4);
}

/* FPSR.QC (bit 27) |= any lane of an i1 vector (or an i1, for the scalar forms) */
static void set_qc(SnxIr* ir, V lanes) {
    V any = LLVMGetTypeKind(LLVMTypeOf(lanes)) == LLVMVectorTypeKind
                ? ir_intrinsic(ir, "llvm.vector.reduce.or", LLVMTypeOf(lanes), &lanes, 1)
                : lanes;
    size_t off = ir_off_fpcr() + 4;
    V fpsr = ir_cpu_load(ir, ir->t_i32, off);
    ir_cpu_store(ir, LLVMBuildOr(B, fpsr, LLVMBuildShl(B, ir_bool32(ir, any), ir_i32(ir, 27), ""), ""), off);
}

/* SQADD, UQADD, SQSUB, UQSUB (vector): saturating, QC when a lane saturates (sqadd_ and friends). For 32/64-bit
   signed lanes on AVX-512 hosts, wunkolo's vpternlog test: a lane overflowed when the inputs' sign bits agree and the
   sum's differs (add, 0x42) or the inputs' differ and the difference's differs from the first (sub, 0x18); the
   saturated value is the result's sign spread and its top bit flipped */
static V saturating(SnxIr* ir, unsigned u, unsigned sub, unsigned eb, unsigned n, V a, V b) {
    static const char* const names[4] = {"llvm.sadd.sat", "llvm.uadd.sat", "llvm.ssub.sat", "llvm.usub.sat"};
    LLVMTypeRef t = LLVMTypeOf(a);
    V wrap = sub ? LLVMBuildSub(B, a, b, "") : LLVMBuildAdd(B, a, b, ""), args[2], r;
    if (!u && eb >= 32 && n * eb == 128 && host_ternlog(ir)) {
        unsigned table = sub ? (TERN_A ^ TERN_B) & (TERN_A ^ TERN_C) : ~(TERN_A ^ TERN_B) & (TERN_A ^ TERN_C);
        V check = LLVMBuildBitCast(B, ternlog(ir, a, b, wrap, table, eb), t, "");
        V overflow = LLVMBuildICmp(B, LLVMIntSLT, check, LLVMConstNull(t), "");
        V spread = LLVMBuildAShr(B, wrap, broadcast(ir, LLVMConstInt(LLVMGetElementType(t), eb - 1, 0), n), "");
        V value = LLVMBuildXor(B, spread,
                               broadcast(ir, LLVMConstInt(LLVMGetElementType(t), (SnxU64)1 << (eb - 1), 0), n), "");
        set_qc(ir, overflow);
        return LLVMBuildSelect(B, overflow, value, wrap, "");
    }
    args[0] = a;
    args[1] = b;
    r = ir_intrinsic(ir, names[sub * 2 + u], t, args, 2); /* PADDS/PADDUS/PSUBS/PSUBUS for 8/16-bit lanes on x86 */
    set_qc(ir, LLVMBuildICmp(B, LLVMIntNE, r, wrap, ""));
    return r;
}

/* SHADD, UHADD, SRHADD, URHADD: (a + b [+ 1]) >> 1 without overflow. Unsigned 8/16-bit rounding is PAVGB/PAVGW on
   x86; signed goes through it too, biased by the lanes' sign bit (wunkolo, "pavgb: most-significant-bit constant"):
   (a ^ 0x80) and (b ^ 0x80) are a + 128 and b + 128, so their average is the signed one plus 128 */
static V halving(SnxIr* ir, unsigned u, unsigned rounding, unsigned size, unsigned n, V a, V b) {
    LLVMTypeRef t = LLVMTypeOf(a), wide = LLVMVectorType(lane_type(ir, size + 1), n);
    V bias = NULL, x, y, r;
    if (!u && size < 2) {
        bias = broadcast(ir, LLVMConstInt(lane_type(ir, size), (SnxU64)1 << ((8u << size) - 1), 0), n);
        a = LLVMBuildXor(B, a, bias, "");
        b = LLVMBuildXor(B, b, bias, "");
        u = 1;
    }
    x = u ? LLVMBuildZExt(B, a, wide, "") : LLVMBuildSExt(B, a, wide, "");
    y = u ? LLVMBuildZExt(B, b, wide, "") : LLVMBuildSExt(B, b, wide, "");
    r = LLVMBuildAdd(B, x, y, "");
    if (rounding) r = LLVMBuildAdd(B, r, broadcast(ir, LLVMConstInt(lane_type(ir, size + 1), 1, 0), n), "");
    r = u ? LLVMBuildLShr(B, r, broadcast(ir, LLVMConstInt(lane_type(ir, size + 1), 1, 0), n), "")
          : LLVMBuildAShr(B, r, broadcast(ir, LLVMConstInt(lane_type(ir, size + 1), 1, 0), n), "");
    r = LLVMBuildTrunc(B, r, t, "");
    return bias ? LLVMBuildXor(B, r, bias, "") : r;
}

/* <n x iN> all-ones or zero per lane, as the compares write them */
static V mask_of(SnxIr* ir, V condition, LLVMTypeRef lanes) { return LLVMBuildSExt(B, condition, lanes, ""); }

/* ---- integer lanes as vectors: the helpers' sx/zx arithmetic in a wider lane where it needs the room ---- */

/* n lanes of 1<<size bytes at base, always a vector (<1 x iN> for the scalar forms, so one emitter serves both) */
static V vlanes(SnxIr* ir, V base, unsigned size, unsigned n) {
    V load = ir_load(ir, LLVMVectorType(lane_type(ir, size), n), base);
    LLVMSetAlignment(load, 1);
    return load;
}
/* the 64-bit half of Vn that Q selects (the "2" forms read the upper one) */
static V half_lanes(SnxIr* ir, unsigned rn, unsigned size, unsigned upper) {
    return vlanes(ir, ir_cpu(ir, ir_off_v(rn) + (upper ? 8 : 0)), size, 8u >> size);
}
static unsigned lanes_of(V x) { return LLVMGetVectorSize(LLVMTypeOf(x)); }
static unsigned bits_of(V x) { return LLVMGetIntTypeWidth(LLVMGetElementType(LLVMTypeOf(x))); }
static V vsplat(SnxIr* ir, V scalar, unsigned n) { /* broadcast, but a vector for one lane too */
    return n == 1 ? LLVMBuildInsertElement(B, LLVMGetUndef(LLVMVectorType(LLVMTypeOf(scalar), 1)), scalar,
                                           ir_i32(ir, 0), "")
                  : broadcast(ir, scalar, n);
}
/* v in every lane of a vector shaped like x */
static V splat_like(SnxIr* ir, V x, SnxU64 v) {
    return vsplat(ir, LLVMConstInt(LLVMGetElementType(LLVMTypeOf(x)), v, 0), lanes_of(x));
}
/* lanes zero- or sign-extended to bits, or truncated to them (LLVM's casts refuse the same width: x itself) */
static V extend(SnxIr* ir, V x, unsigned u, unsigned bits) {
    LLVMTypeRef t = LLVMVectorType(LLVMIntTypeInContext(ir->ctx, bits), lanes_of(x));
    if (bits == bits_of(x)) return x;
    if (bits < bits_of(x)) return LLVMBuildTrunc(B, x, t, "");
    return u ? LLVMBuildZExt(B, x, t, "") : LLVMBuildSExt(B, x, t, "");
}
static V widen(SnxIr* ir, V x, unsigned u) { return extend(ir, x, u, 2 * bits_of(x)); }
static V narrow(SnxIr* ir, V x) {
    return LLVMBuildTrunc(B, x, LLVMVectorType(LLVMIntTypeInContext(ir->ctx, bits_of(x) / 2), lanes_of(x)), "");
}
static SnxU64 int_min(unsigned eb) { return (SnxU64)1 << (eb - 1); }
static SnxU64 int_max(unsigned eb) { return int_min(eb) - 1; }
static SnxU64 uint_max(unsigned eb) { return eb == 64 ? ~(SnxU64)0 : ((SnxU64)1 << eb) - 1; }

/* the narrowing forms' result: Vd's low half with the upper half zero, or (the "2" forms) its upper half with the
   low half kept (simd_2misc's and simd_shimm's o = Q ? d : r) */
static void write_narrow(SnxIr* ir, unsigned rd, V value, unsigned upper) {
    if (upper)
        set_lanes(ir, ir_cpu(ir, ir_off_v(rd) + 8), value);
    else
        write_lanes(ir, rd, value, 0);
}

/* wide lanes clamped to half their width, QC for any lane that changed: signed to signed (ssat, SQXTN), signed to
   unsigned (usat, SQXTUN) or unsigned to unsigned (UQXTN) */
static V saturate_narrow(SnxIr* ir, V x, unsigned from_signed, unsigned to_signed) {
    unsigned eb = bits_of(x) / 2;
    V hi = splat_like(ir, x, to_signed ? int_max(eb) : uint_max(eb)), over, under, r;
    over = LLVMBuildICmp(B, from_signed ? LLVMIntSGT : LLVMIntUGT, x, hi, "");
    r = LLVMBuildSelect(B, over, hi, x, "");
    if (from_signed) { /* INT_MIN sign-extended, for lanes of 128 bits too (SUQADD's) */
        V lo = to_signed ? vsplat(ir, LLVMConstInt(LLVMGetElementType(LLVMTypeOf(x)), (SnxU64)0 - int_min(eb), 1),
                                  lanes_of(x))
                         : splat_like(ir, x, 0);
        under = LLVMBuildICmp(B, LLVMIntSLT, x, lo, "");
        r = LLVMBuildSelect(B, under, lo, r, "");
        over = LLVMBuildOr(B, over, under, "");
    }
    set_qc(ir, over);
    return narrow(ir, r);
}

/* SQDMULL and the doubling in SQDMLAL/SQDMLSL: 2*a*b in twice the width. Only INT_MIN * INT_MIN overflows it, and
   saturates with QC */
static V doubling_long(SnxIr* ir, V a, V b) {
    unsigned eb = bits_of(a);
    V x = widen(ir, a, 0), p = LLVMBuildShl(B, LLVMBuildMul(B, x, widen(ir, b, 0), ""), splat_like(ir, x, 1), "");
    V min = splat_like(ir, a, int_min(eb));
    V both = LLVMBuildAnd(B, LLVMBuildICmp(B, LLVMIntEQ, a, min, ""), LLVMBuildICmp(B, LLVMIntEQ, b, min, ""), "");
    set_qc(ir, both);
    return LLVMBuildSelect(B, both, splat_like(ir, p, int_max(2 * eb)), p, "");
}
/* SQDMULH, SQRDMULH: the high half of 2*a*b (plus 1<<(eb-1) rounding), computed in twice the width where it cannot
   overflow but for INT_MIN * INT_MIN, which saturates with QC */
static V doubling_high(SnxIr* ir, V a, V b, unsigned round) {
    unsigned eb = bits_of(a);
    V x = widen(ir, a, 0), p, min, both;
    p = LLVMBuildShl(B, LLVMBuildMul(B, x, widen(ir, b, 0), ""), splat_like(ir, x, 1), "");
    if (round) p = LLVMBuildAdd(B, p, splat_like(ir, x, int_min(eb)), "");
    p = narrow(ir, LLVMBuildAShr(B, p, splat_like(ir, x, eb), ""));
    min = splat_like(ir, a, int_min(eb));
    both = LLVMBuildAnd(B, LLVMBuildICmp(B, LLVMIntEQ, a, min, ""), LLVMBuildICmp(B, LLVMIntEQ, b, min, ""), "");
    set_qc(ir, both);
    return LLVMBuildSelect(B, both, splat_like(ir, a, int_max(eb)), p, "");
}

/* PMUL, PMULL (bytes): carry-less products of byte lanes, 16 bits wide: b shifted by each set bit of a, xored */
static V poly_multiply(SnxIr* ir, V a, V b) {
    V x = widen(ir, a, 1), y = widen(ir, b, 1), p = splat_like(ir, x, 0), bit;
    unsigned k;
    for (k = 0; k < 8; ++k) {
        bit = LLVMBuildAnd(B, LLVMBuildLShr(B, x, splat_like(ir, x, k), ""), splat_like(ir, x, 1), "");
        p = LLVMBuildXor(
            B, p, LLVMBuildAnd(B, LLVMBuildNeg(B, bit, ""), LLVMBuildShl(B, y, splat_like(ir, y, k), ""), ""), "");
    }
    return p;
}

/* the long forms on narrow lanes a and b, by three-different's opcode: [SU]ADDL (0), [SU]SUBL (2), [SU]ABAL (5),
   [SU]ABDL (7), [SU]MLAL (8), SQDMLAL (9), [SU]MLSL (10), SQDMLSL (11), [SU]MULL (12), SQDMULL (13); z the
   accumulator's wide lanes. Every result fits the wide lane, so the helper's 64-bit sums truncate to the same bits */
static V long_op(SnxIr* ir, unsigned op, unsigned u, V a, V b, V z) {
    V x, y, r;
    if (op == 9 || op == 11 || op == 13) {
        r = doubling_long(ir, a, b);
        return op == 13 ? r : saturating(ir, 0, op == 11, 2 * bits_of(a), lanes_of(a), z, r);
    }
    x = widen(ir, a, u);
    y = widen(ir, b, u);
    switch (op) {
        case 0: return LLVMBuildAdd(B, x, y, "");
        case 2: return LLVMBuildSub(B, x, y, "");
        case 5:
        case 7:
            r = LLVMBuildSelect(B, LLVMBuildICmp(B, u ? LLVMIntUGT : LLVMIntSGT, a, b, ""), LLVMBuildSub(B, x, y, ""),
                                LLVMBuildSub(B, y, x, ""), "");
            return op == 5 ? LLVMBuildAdd(B, z, r, "") : r;
        case 8: return LLVMBuildAdd(B, z, LLVMBuildMul(B, x, y, ""), "");
        case 10: return LLVMBuildSub(B, z, LLVMBuildMul(B, x, y, ""), "");
        default: return LLVMBuildMul(B, x, y, "");
    }
}

/* SSHL, USHL, SRSHL, URSHL, SQSHL, UQSHL, SQRSHL, UQRSHL (register): vshl (helpers/simd.c) on 64-bit copies of the
   lanes, the shift the signed low byte of b's lane. Every shift amount is clamped below 64 and the cases it would
   change are selected away: LLVM's shifts by the width or more are poison. QC lanes go to *saturated */
static V register_shift(SnxIr* ir, V a, V b, unsigned u, unsigned round, unsigned sat) {
    unsigned eb = bits_of(a);
    V x = extend(ir, a, u, 64), sh = extend(ir, extend(ir, b, 1, 8), 0, 64);
    V zero = splat_like(ir, x, 0), k63 = splat_like(ir, x, 63), keb = splat_like(ir, x, eb), left, right, r;
    V amount, shifted, fits, negative, back;
    /* left: sh >= 0 */
    amount = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntUGT, sh, k63, ""), k63, sh, "");
    shifted = LLVMBuildShl(B, x, amount, "");
    if (!sat)
        left = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntSGE, sh, keb, ""), zero, shifted, "");
    else {
        /* no bit lost: shifting back gives x, and the result fits the lane (vshl's checks) */
        back = u ? LLVMBuildLShr(B, shifted, amount, "") : LLVMBuildAShr(B, shifted, amount, "");
        fits = LLVMBuildICmp(B, LLVMIntEQ, extend(ir, extend(ir, shifted, u, eb), u, 64), shifted, "");
        fits = LLVMBuildAnd(B, fits, LLVMBuildICmp(B, LLVMIntEQ, back, x, ""), "");
        fits = LLVMBuildAnd(B, fits, LLVMBuildICmp(B, LLVMIntSLT, sh, keb, ""), "");
        fits = LLVMBuildOr(B, fits, LLVMBuildICmp(B, LLVMIntEQ, x, zero, ""), "");
        left = LLVMBuildSelect(B, fits, shifted,
                               u ? splat_like(ir, x, uint_max(eb))
                                 : LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntSLT, x, zero, ""),
                                                   splat_like(ir, x, int_min(eb)), splat_like(ir, x, int_max(eb)), ""),
                               "");
        set_qc(ir, LLVMBuildAnd(B, LLVMBuildICmp(B, LLVMIntSGE, sh, zero, ""), LLVMBuildNot(B, fits, ""), ""));
    }
    /* right: n = -sh, 1..128; a shift by 64 or more leaves the sign (signed) or 0 */
    negative = LLVMBuildSub(B, zero, sh, "");
    amount = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntUGT, negative, k63, ""), k63, negative, "");
    if (u)
        right = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntUGT, negative, k63, ""), zero,
                                LLVMBuildLShr(B, x, amount, ""), "");
    else
        right = LLVMBuildAShr(B, x, amount, "");
    if (round) { /* + the last bit shifted out, bit n-1; past a shift of 64 there is nothing left */
        V last = LLVMBuildSub(B, negative, splat_like(ir, x, 1), "");
        V bit;
        last = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntUGT, last, k63, ""), k63, last, "");
        bit = LLVMBuildAnd(B, LLVMBuildLShr(B, x, last, ""), splat_like(ir, x, 1), "");
        right = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntUGT, negative, splat_like(ir, x, 64), ""), zero,
                                LLVMBuildAdd(B, right, bit, ""), "");
    }
    r = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntSLT, sh, zero, ""), right, left, "");
    return extend(ir, r, 1, eb);
}

/* three-same integer (vector): simd_3same for BSL/BIT/BIF, compares, MIN/MAX, ABD/ABA, MLA/MLS, MUL, pairwise
   ADDP/MAXP/MINP */
static int three_same(SnxIr* ir, SnxU32 w) {
    unsigned q = (w >> 30) & 1, u = (w >> 29) & 1, size = (w >> 22) & 3, rm = (w >> 16) & 31, op = (w >> 11) & 31,
             rn = (w >> 5) & 31, rd = w & 31, eb = 8u << size, n, i;
    LLVMTypeRef t;
    V a, b, z, r;
    if (op == 3) { /* BSL, BIT, BIF (fast_simd has the others): one VPTERNLOGQ on AVX-512 hosts */
        LLVMTypeRef t = LLVMVectorType(ir->t_i64, 2);
        V x, y, d, r;
        if (!u || !size) return 0;
        x = ir_load(ir, t, vreg(ir, rn));
        y = ir_load(ir, t, vreg(ir, rm));
        d = ir_load(ir, t, vreg(ir, rd));
        LLVMSetAlignment(x, 8); /* V128 is 8-aligned in the CPU structure */
        LLVMSetAlignment(y, 8);
        LLVMSetAlignment(d, 8);
        if (host_ternlog(ir)) {
            unsigned table = size == 1   ? (TERN_C & TERN_A) | (~TERN_C & TERN_B)  /* BSL: (d & n) | (~d & m) */
                             : size == 2 ? (TERN_C & ~TERN_B) | (TERN_A & TERN_B)  /* BIT: (d & ~m) | (n & m) */
                                         : (TERN_C & TERN_B) | (TERN_A & ~TERN_B); /* BIF: (d & m) | (n & ~m) */
            r = LLVMBuildBitCast(B, ternlog(ir, x, y, d, table, 64), t, "");
        } else if (size == 1)
            r = LLVMBuildOr(B, LLVMBuildAnd(B, d, x, ""), LLVMBuildAnd(B, LLVMBuildNot(B, d, ""), y, ""), "");
        else if (size == 2)
            r = LLVMBuildOr(B, LLVMBuildAnd(B, d, LLVMBuildNot(B, y, ""), ""), LLVMBuildAnd(B, x, y, ""), "");
        else
            r = LLVMBuildOr(B, LLVMBuildAnd(B, d, y, ""), LLVMBuildAnd(B, x, LLVMBuildNot(B, y, ""), ""), "");
        if (!q) r = LLVMBuildInsertElement(B, r, ir_i64(ir, 0), ir_i32(ir, 1), "");
        LLVMSetAlignment(LLVMBuildStore(B, r, vreg(ir, rd)), 8);
        return 1;
    }
    if (op >= 0x18 || (size == 3 && !q)) return 0;
    n = (q ? 128u : 64u) / eb;
    t = LLVMVectorType(lane_type(ir, size), n);
    a = get_lanes(ir, vreg(ir, rn), size, n);
    b = get_lanes(ir, vreg(ir, rm), size, n);
    switch (op) {
        case 0x00:
        case 0x02:
            if (eb == 64) return 0;
            r = halving(ir, u, op == 0x02, size, n, a, b);
            break;
        case 0x01:
        case 0x05: r = saturating(ir, u, op == 0x05, eb, n, a, b); break;
        case 0x04: /* SHSUB, UHSUB: the difference in twice the width, halved arithmetically */
            if (eb == 64) return 0;
            r = narrow(ir, LLVMBuildAShr(B, LLVMBuildSub(B, widen(ir, a, u), widen(ir, b, u), ""),
                                         splat_like(ir, widen(ir, a, u), 1), ""));
            break;
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: r = register_shift(ir, a, b, u, op == 0x0A || op == 0x0B, op == 0x09 || op == 0x0B); break;
        case 0x16:
            if (size != 1 && size != 2) return 0;
            r = doubling_high(ir, a, b, u);
            break;
        case 0x06: r = mask_of(ir, LLVMBuildICmp(B, u ? LLVMIntUGT : LLVMIntSGT, a, b, ""), t); break;
        case 0x07: r = mask_of(ir, LLVMBuildICmp(B, u ? LLVMIntUGE : LLVMIntSGE, a, b, ""), t); break;
        case 0x11:
            r = u ? mask_of(ir, LLVMBuildICmp(B, LLVMIntEQ, a, b, ""), t)
                  : mask_of(ir, LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAnd(B, a, b, ""), LLVMConstNull(t), ""), t);
            break;
        case 0x0C:
        case 0x0D:
            if (eb == 64) return 0;
            r = LLVMBuildSelect(
                B,
                LLVMBuildICmp(B, op == 0x0C ? (u ? LLVMIntUGT : LLVMIntSGT) : (u ? LLVMIntULT : LLVMIntSLT), a, b, ""),
                a, b, "");
            break;
        case 0x0E:
        case 0x0F:
            if (eb == 64) return 0;
            r = LLVMBuildSelect(B, LLVMBuildICmp(B, u ? LLVMIntUGT : LLVMIntSGT, a, b, ""), LLVMBuildSub(B, a, b, ""),
                                LLVMBuildSub(B, b, a, ""), "");
            if (op == 0x0F) r = LLVMBuildAdd(B, get_lanes(ir, vreg(ir, rd), size, n), r, "");
            break;
        case 0x12:
            if (eb == 64) return 0;
            z = get_lanes(ir, vreg(ir, rd), size, n);
            r = u ? LLVMBuildSub(B, z, LLVMBuildMul(B, a, b, ""), "")
                  : LLVMBuildAdd(B, z, LLVMBuildMul(B, a, b, ""), "");
            break;
        case 0x13: /* MUL, PMUL (bytes) */
            if (eb == 64 || (u && size)) return 0;
            r = u ? narrow(ir, poly_multiply(ir, a, b)) : LLVMBuildMul(B, a, b, "");
            break;
        case 0x14:
        case 0x15:
        case 0x17: {
            /* pairwise: lanes 0..n/2-1 from pairs of a, the rest from pairs of b */
            unsigned even[16], odd[16];
            V x, y;
            if ((op != 0x17 && eb == 64) || (op == 0x17 && u)) return 0;
            for (i = 0; i < n; ++i) {
                even[i] = i < n / 2 ? 2 * i : n + 2 * (i - n / 2);
                odd[i] = even[i] + 1;
            }
            x = shuffle(ir, a, b, even, n);
            y = shuffle(ir, a, b, odd, n);
            if (op == 0x17)
                r = LLVMBuildAdd(B, x, y, "");
            else
                r = LLVMBuildSelect(
                    B,
                    LLVMBuildICmp(B, op == 0x14 ? (u ? LLVMIntUGT : LLVMIntSGT) : (u ? LLVMIntULT : LLVMIntSLT), x, y,
                                  ""),
                    x, y, "");
            break;
        }
        default: return 0;
    }
    write_lanes(ir, rd, r, q);
    return 1;
}

/* byte-lane shifts by immediate through GF2P8AFFINEQB: simd_shimm's semantics for eb 8 */
static int shift_bytes_gfni(SnxIr* ir, unsigned q, unsigned u, unsigned op, unsigned shr, unsigned shl, unsigned rn,
                            unsigned rd) {
    unsigned n = q ? 16 : 8;
    V a = v128_type_load(ir, vreg(ir, rn)), r, z;
    switch (op) {
        case 0x00:
        case 0x02:
        case 0x04:
        case 0x06: /* [SU][R]SHR, [SU][R]SRA */
            r = gf_affine(ir, a, u ? gf_lshr(shr) : gf_ashr(shr));
            if (op & 4) /* + the last bit shifted out (signed or not, it is source bit shr-1) */
                r = LLVMBuildAdd(B, r, gf_affine(ir, a, gf_bit(shr - 1)), "");
            if (op & 2) r = LLVMBuildAdd(B, v128_type_load(ir, vreg(ir, rd)), r, "");
            break;
        case 0x08: /* SRI */
            if (!u) return 0;
            z = v128_type_load(ir, vreg(ir, rd));
            if (shr >= 8)
                r = z;
            else
                r = LLVMBuildOr(B, LLVMBuildAnd(B, z, broadcast(ir, ir_i8(ir, ~(0xFFu >> shr)), 16), ""),
                                gf_affine(ir, a, gf_lshr(shr)), "");
            break;
        case 0x0A:
            r = gf_affine(ir, a, gf_shl(shl));
            if (u) { /* SLI */
                z = v128_type_load(ir, vreg(ir, rd));
                r = LLVMBuildOr(B, LLVMBuildAnd(B, z, broadcast(ir, ir_i8(ir, ~(0xFFu << shl)), 16), ""), r, "");
            }
            break;
        default: return 0;
    }
    write_lanes(ir, rd, low_lanes(ir, r, n), q);
    return 1;
}

/* the narrowing shifts right on the wide lanes of Vn, shr 1..eb: SHRN and RSHRN wrap, the saturating ones clamp
   with QC (and have scalar forms: one lane, the rest of Vd zero). Rounding adds the last bit shifted out (simd_shimm:
   the constant first would overflow 64 bits) */
static int shift_narrow(SnxIr* ir, unsigned q, unsigned u, unsigned op, unsigned size, unsigned shr, unsigned rn,
                        unsigned rd, unsigned scalar) {
    unsigned rnd = op & 1;
    V x, v, bit;
    if (scalar && !u && op < 0x12) return 0;
    x = vlanes(ir, vreg(ir, rn), size + 1, scalar ? 1 : 8u >> size);
    if (!u && op < 0x12) { /* SHRN, RSHRN: the bits kept end below the wide lane's top, so its carry is not needed */
        if (rnd) x = LLVMBuildAdd(B, x, splat_like(ir, x, (SnxU64)1 << (shr - 1)), "");
        write_narrow(ir, rd, narrow(ir, LLVMBuildLShr(B, x, splat_like(ir, x, shr), "")), q);
        return 1;
    }
    /* U with op 0x12/0x13: UQSHRN, UQRSHRN (unsigned); else SQSHRN, SQRSHRN, SQSHRUN, SQRSHRUN (signed source) */
    v = u && op >= 0x12 ? LLVMBuildLShr(B, x, splat_like(ir, x, shr), "")
                        : LLVMBuildAShr(B, x, splat_like(ir, x, shr), "");
    if (rnd) {
        bit = LLVMBuildAnd(B, LLVMBuildLShr(B, x, splat_like(ir, x, shr - 1), ""), splat_like(ir, x, 1), "");
        v = LLVMBuildAdd(B, v, bit, "");
    }
    write_narrow(ir, rd, saturate_narrow(ir, v, !(u && op >= 0x12), !u), q);
    return 1;
}

/* shift by immediate: simd_shimm's SSHR/USHR, SSRA/USRA, SRSHR/URSHR, SRSRA/URSRA, SHL, SLI, SRI, SQSHL/UQSHL,
   SQSHLU, SSHLL/USHLL and the narrowing shifts; scalar, the 64-bit forms and the saturating left shifts of any size
   (as simd_shimm allows) */
static int shift_immediate(SnxIr* ir, SnxU32 w, unsigned scalar) {
    unsigned q = scalar ? 0 : (w >> 30) & 1, u = (w >> 29) & 1, immh = (w >> 19) & 15, immb = (w >> 16) & 7,
             op = (w >> 11) & 31, rn = (w >> 5) & 31, rd = w & 31, size = 3, eb, imm, shr, shl, n;
    LLVMTypeRef t;
    V a, r;
    if (!immh) return 0;
    while (!((immh >> size) & 1)) size--;
    eb = 8u << size;
    imm = (immh << 3) | immb;
    shr = 2 * eb - imm;
    shl = imm - eb;
    if (op == 0x14 || (op >= 0x10 && op <= 0x13)) {
        if (size == 3) return 0;
        if (op != 0x14) return shift_narrow(ir, q, u, op, size, shr, rn, rd, scalar);
        if (scalar) return 0;
        /* SSHLL, USHLL: the low or high half widened, shifted left */
        r = widen(ir, half_lanes(ir, rn, size, q), u);
        set_lanes(ir, vreg(ir, rd), LLVMBuildShl(B, r, splat_like(ir, r, shl), ""));
        return 1;
    }
    if (size == 3 && !q && !scalar) return 0;
    if (scalar && size != 3 && op != 0x0C && op != 0x0E) return 0;
    if (!size && !scalar && host_gfni(ir) && shift_bytes_gfni(ir, q, u, op, shr, shl, rn, rd)) return 1;
    n = scalar ? 1 : (q ? 128u : 64u) / eb;
    t = lanes_type(ir, size, n);
    a = get_lanes(ir, vreg(ir, rn), size, n);
#define SPLAT(v) broadcast(ir, LLVMConstInt(lane_type(ir, size), (v), 0), n)
    switch (op) {
        case 0x00:
        case 0x02:
        case 0x04:
        case 0x06: /* [SU][R]SHR, [SU][R]SRA: a shift by the element size gives 0 (unsigned) or the sign */
            if (u)
                r = shr >= eb ? LLVMConstNull(t) : LLVMBuildLShr(B, a, SPLAT(shr), "");
            else
                r = LLVMBuildAShr(B, a, SPLAT(shr >= eb ? eb - 1 : shr), "");
            if (op & 4) { /* rounding: + bit shr-1 of the source (for shr = eb, 0 signed, the top bit unsigned) */
                V bit = LLVMBuildAnd(B, LLVMBuildLShr(B, a, SPLAT(shr - 1), ""), SPLAT(1), "");
                r = shr >= eb && !u ? LLVMConstNull(t) : LLVMBuildAdd(B, r, bit, "");
            }
            if (op & 2) r = LLVMBuildAdd(B, get_lanes(ir, vreg(ir, rd), size, n), r, "");
            break;
        case 0x08: /* SRI */
            if (!u) return 0;
            if (shr >= eb)
                r = get_lanes(ir, vreg(ir, rd), size, n);
            else {
                SnxU64 m = (eb == 64 ? ~(SnxU64)0 : ((SnxU64)1 << eb) - 1) >> shr;
                r = LLVMBuildOr(B, LLVMBuildAnd(B, get_lanes(ir, vreg(ir, rd), size, n), SPLAT(~m), ""),
                                LLVMBuildLShr(B, a, SPLAT(shr), ""), "");
            }
            break;
        case 0x0A:
            if (!u) /* SHL */
                r = LLVMBuildShl(B, a, SPLAT(shl), "");
            else { /* SLI */
                SnxU64 m = (eb == 64 ? ~(SnxU64)0 : ((SnxU64)1 << eb) - 1) << shl;
                r = LLVMBuildOr(B, LLVMBuildAnd(B, get_lanes(ir, vreg(ir, rd), size, n), SPLAT(~m), ""),
                                LLVMBuildAnd(B, LLVMBuildShl(B, a, SPLAT(shl), ""), SPLAT(m), ""), "");
            }
            break;
        case 0x0C:   /* SQSHLU: a negative lane gives 0, then as UQSHL */
        case 0x0E: { /* SQSHL, UQSHL: saturated when shifting back does not give the lane (no bit lost, no sign
                        change), QC */
            V shifted = LLVMBuildShl(B, a, SPLAT(shl), ""), lost, negative = NULL, top;
            if (op == 0x0C && !u) return 0;
            if (!u) {
                lost = LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAShr(B, shifted, SPLAT(shl), ""), a, "");
                top = LLVMBuildSelect(B, LLVMBuildICmp(B, LLVMIntSLT, a, LLVMConstNull(t), ""), SPLAT(int_min(eb)),
                                      SPLAT(int_max(eb)), "");
            } else {
                lost = LLVMBuildICmp(B, LLVMIntNE, LLVMBuildLShr(B, shifted, SPLAT(shl), ""), a, "");
                top = SPLAT(uint_max(eb));
            }
            r = LLVMBuildSelect(B, lost, top, shifted, "");
            if (op == 0x0C) {
                negative = LLVMBuildICmp(B, LLVMIntSLT, a, LLVMConstNull(t), "");
                r = LLVMBuildSelect(B, negative, LLVMConstNull(t), r, "");
                lost = LLVMBuildOr(B, lost, negative, "");
            }
            set_qc(ir, lost);
            break;
        }
        default: return 0;
    }
#undef SPLAT
    write_lanes(ir, rd, r, q);
    return 1;
}

