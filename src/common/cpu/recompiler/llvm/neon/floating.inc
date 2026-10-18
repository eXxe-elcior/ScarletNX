/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
/* FABS, FNEG (vector): sign bits only */
static int fp_sign_misc(SnxIr* ir, SnxU32 w) {
    unsigned q = (w >> 30) & 1, u = (w >> 29) & 1, size = (w >> 22) & 3, op = (w >> 12) & 31, rn = (w >> 5) & 31,
             rd = w & 31, sz = size & 1;
    SnxU64 sign = sz ? (SnxU64)1 << 63 : ((SnxU64)0x80000000UL << 32) | 0x80000000UL;
    V lo, hi;
    if (op != 0x0F || !(size >> 1) || (sz && !q)) return 0;
    lo = ir_vd(ir, rn, 0);
    hi = ir_vd(ir, rn, 1);
    if (u) {
        lo = LLVMBuildXor(B, lo, ir_i64(ir, sign), "");
        hi = LLVMBuildXor(B, hi, ir_i64(ir, sign), "");
    } else {
        lo = LLVMBuildAnd(B, lo, ir_i64(ir, ~sign), "");
        hi = LLVMBuildAnd(B, hi, ir_i64(ir, ~sign), "");
    }
    ir_set_vd(ir, rd, 0, lo);
    ir_set_vd(ir, rd, 1, q ? hi : ir_i64(ir, 0));
    return 1;
}

typedef enum {
    FP_LANES_ADD,
    FP_LANES_ABD,
    FP_LANES_MULX,
    FP_LANES_STEP,
    FP_LANES_MINMAX,
    FP_LANES_COMPARE,
    FP_LANES_SQRT,
    FP_LANES_ESTIMATE,
    FP_LANES_TO_INT,
    FP_LANES_FROM_INT
} FpLaneKind;
/* The decoded shape is separate from these operation-specific arguments.
 * Never overload 'arg' and 'flag' with unrelated operation semantics.
 * All members are resolved by the translator, not dispatched at runtime. */
typedef struct {
    FpLaneKind kind;
    union {
        struct { unsigned minimum, numeric; } extremum;
        struct { unsigned predicate; } compare;
        struct { unsigned rsqrt; } estimate;
        struct { unsigned rounding, is_unsigned, fractional_bits; } conversion;
    } p;
} FpOp;

/* one lane through the helper simd_3same's and simd_2misc's loops call (helpers/simd.c), as iN */
static V fp_lane(SnxIr* ir, const FpOp* op, V a, V b, unsigned width) {
    switch (op->kind) {
        case FP_LANES_ADD: return ir_fp_binary(ir, a, b, width, 0);                /* SNX_FP_ADD */
        case FP_LANES_ABD: return ir_fp_binary(ir, a, b, width, 5);                /* SNX_FP_ABD */
        case FP_LANES_MULX: return ir_fp_binary(ir, a, b, width, 4);               /* SNX_FP_MULX */
        case FP_LANES_STEP: return ir_fp_binary(ir, a, b, width, op->p.estimate.rsqrt ? 7 : 6); /* SNX_FP_RSQRTS, SNX_FP_RECPS */
        case FP_LANES_MINMAX: return ir_fp_minmax(ir, a, b, width, op->p.extremum.minimum, op->p.extremum.numeric);
        case FP_LANES_COMPARE: return ir_fp_compare_mask(ir, op->p.compare.predicate, a, b, width);
        case FP_LANES_SQRT: return ir_fp_unary(ir, a, width, 0);                   /* SNX_FP_SQRT */
        case FP_LANES_ESTIMATE: return ir_fp_unary(ir, a, width, op->p.estimate.rsqrt ? 2 : 1); /* SNX_FP_RSQRTE, SNX_FP_RECPE */
        case FP_LANES_TO_INT: return ir_fp_to_int(ir, a, width, width, op->p.conversion.is_unsigned, op->p.conversion.fractional_bits, op->p.conversion.rounding);
        default: return ir_fp_from_int(ir, a, width, width, op->p.conversion.is_unsigned, op->p.conversion.fractional_bits);
    }
}

/* the operation on n lanes of 1<<s bytes (x, and y for two operands) into Vd: every lane at once on the host's
   vector unit when the fast path's guard holds for all of them (ir_fp.c's fast paths), otherwise the helper a lane
   at a time. One lane is a scalar form: the rest of Vd is zero */
static void fp_lanes(SnxIr* ir, const FpOp* op, unsigned s, unsigned n, V x, V y, unsigned rd, unsigned q) {
    unsigned width = 8u << s, i;
    SnxFpPath path;
    SnxFpFast f;
    V r;
    if (op->kind == FP_LANES_ADD || op->kind == FP_LANES_SQRT) {
        SnxFpArithmetic spec = {
            op->kind == FP_LANES_ADD ? SNX_FP_ARITH_ADD : SNX_FP_ARITH_SQRT,
            width, n, x, op->kind == FP_LANES_ADD ? y : NULL, NULL
        };
        write_lanes(ir, rd, snx_fp_emit_arithmetic(ir, &spec), q);
        return;
    }
    switch (op->kind) {
        case FP_LANES_ABD: /* |x - y|: snx_fp_binary clears the difference's sign */
            ir_fp_fast(ir, SNX_IR_FP_SUB, width, n, x, y, NULL, &f);
            f.value = LLVMBuildAnd(
                B, f.value, broadcast(ir, LLVMConstInt(lane_type(ir, s), ((SnxU64)1 << (width - 1)) - 1, 0), n), "");
            break;
        case FP_LANES_MULX: /* FMUL's result but for infinity times zero, which the guard keeps out */
            ir_fp_fast(ir, SNX_IR_FP_MUL, width, n, x, y, NULL, &f);
            break;
        case FP_LANES_STEP: ir_fp_step_fast(ir, op->p.estimate.rsqrt, width, n, x, y, &f); break;
        case FP_LANES_MINMAX: ir_fp_minmax_fast(ir, width, n, x, y, op->p.extremum.minimum, &f); break;
        case FP_LANES_COMPARE: ir_fp_compare_fast(ir, op->p.compare.predicate, width, n, x, y, &f); break;
        case FP_LANES_ESTIMATE: ir_fp_estimate_fast(ir, op->p.estimate.rsqrt, width, n, x, &f); break;
        case FP_LANES_TO_INT:
            if (!ir_fp_to_int_fast(ir, width, n, x, width, op->p.conversion.is_unsigned, op->p.conversion.rounding, op->p.conversion.fractional_bits, &f)) f.ok = NULL;
            break;
        default: ir_fp_from_int_fast(ir, width, n, x, op->p.conversion.is_unsigned, op->p.conversion.fractional_bits, &f); break;
    }
    if (f.ok) { /* else no fast path: the helper alone */
        path = snx_fp_path_begin(ir, f.ok);
        snx_fp_path_else(ir, &path, f.inexact);
    }
    r = n == 1 ? NULL : LLVMGetUndef(LLVMTypeOf(x));
    for (i = 0; i < n; ++i) {
        V a = n == 1 ? x : LLVMBuildExtractElement(B, x, ir_i32(ir, i), ""), b = NULL, v;
        if (y) b = n == 1 ? y : LLVMBuildExtractElement(B, y, ir_i32(ir, i), "");
        v = ir_cast(ir, fp_lane(ir, op, a, b, width), lane_type(ir, s), 0);
        r = n == 1 ? v : LLVMBuildInsertElement(B, r, v, ir_i32(ir, i), "");
    }
    if (f.ok) r = snx_fp_path_merge(ir, &path, f.value, r);
    write_lanes(ir, rd, r, q);
}

/* FP three-same beyond fast_simd's arithmetic: simd_3same's FADDP, FABD, FMULX, FRECPS, FRSQRTS, FMAX/FMIN/FMAXNM/
   FMINNM and their pairwise forms, FCMEQ/FCMGE/FCMGT and FACGE/FACGT; the scalar forms of FABD, FMULX, FRECPS,
   FRSQRTS and the compares (simd_3same refuses the scalar pairwise, accumulating and min/max ones) */
static int fp_three_same(SnxIr* ir, SnxU32 w, unsigned scalar) {
    unsigned q = scalar ? 0 : (w >> 30) & 1, u = (w >> 29) & 1, sz = (w >> 22) & 1, hi = (w >> 23) & 1,
             rm = (w >> 16) & 31, op = (w >> 11) & 31, rn = (w >> 5) & 31, rd = w & 31, s = sz ? 3 : 2,
             n = scalar ? 1
                 : sz   ? 2
                        : (q ? 4 : 2),
             i;
    unsigned pair = 0, absolute = 0;
    FpOp o;
    V x, y;
    if (!scalar && sz && !q) return 0;
    memset(&o, 0, sizeof(o));
    o.p.extremum.minimum = hi;
    o.p.extremum.numeric = op == 0x18;
    switch (op) {
        case 0x18: /* FMAXNM, FMINNM */
        case 0x1E: /* FMAX, FMIN; with U, the pairwise FMAXNMP, FMINNMP, FMAXP, FMINP */
            o.kind = FP_LANES_MINMAX;
            pair = u;
            break;
        case 0x1A: /* FADDP, FABD (FADD and FSUB are fast_simd's) */
            if (!u) return 0;
            o.kind = hi ? FP_LANES_ABD : FP_LANES_ADD;
            pair = !hi;
            break;
        case 0x1B: /* FMULX (FMUL is fast_simd's) */
            if (u || hi) return 0;
            o.kind = FP_LANES_MULX;
            break;
        case 0x1F: /* FRECPS, FRSQRTS (FDIV is fast_simd's) */
            if (u) return 0;
            o.kind = FP_LANES_STEP;
            break;
        case 0x1C:
            if (!u && hi) return 0;
            o.kind = FP_LANES_COMPARE;
            o.p.compare.predicate = !u ? SNX_IR_CMP_EQ : hi ? SNX_IR_CMP_GT : SNX_IR_CMP_GE;
            break;
        case 0x1D: /* FACGE, FACGT: the comparison of the magnitudes */
            if (!u) return 0;
            o.kind = FP_LANES_COMPARE;
            o.p.compare.predicate = hi ? SNX_IR_CMP_GT : SNX_IR_CMP_GE;
            absolute = 1;
            break;
        default: return 0;
    }
    if (scalar && o.kind != FP_LANES_ABD && o.kind != FP_LANES_MULX && o.kind != FP_LANES_STEP &&
        o.kind != FP_LANES_COMPARE)
        return 0;
    x = get_lanes(ir, vreg(ir, rn), s, n);
    y = get_lanes(ir, vreg(ir, rm), s, n);
    if (pair) { /* lanes 0..n/2-1 from pairs of Vn, the rest from pairs of Vm: in Vn:Vm, elements 2i and 2i+1 */
        unsigned even[4], odd[4];
        V a = x;
        for (i = 0; i < n; ++i) {
            even[i] = 2 * i;
            odd[i] = 2 * i + 1;
        }
        x = shuffle(ir, a, y, even, n);
        y = shuffle(ir, a, y, odd, n);
    }
    if (absolute) {
        V magnitude = broadcast(ir, LLVMConstInt(lane_type(ir, s), sz ? ~((SnxU64)1 << 63) : 0x7FFFFFFFUL, 0), n);
        x = LLVMBuildAnd(B, x, magnitude, "");
        y = LLVMBuildAnd(B, y, magnitude, "");
    }
    fp_lanes(ir, &o, s, n, x, y, rd, q);
    return 1;
}

/* FP two-register misc: simd_2misc's FCMGT, FCMGE, FCMEQ, FCMLE, FCMLT #0.0, FCVT[NPMZA][SU], SCVTF, UCVTF, FRECPE
   and FRSQRTE (vector and scalar) and FSQRT (vector) */
static int fp_misc(SnxIr* ir, SnxU32 w, unsigned scalar) {
    unsigned q = scalar ? 0 : (w >> 30) & 1, u = (w >> 29) & 1, size = (w >> 22) & 3, op = (w >> 12) & 31,
             rn = (w >> 5) & 31, rd = w & 31, sz = size & 1, s = sz ? 3 : 2,
             n = scalar ? 1
                 : sz   ? 2
                        : (q ? 4 : 2);
    unsigned k = u << 6 | (size >> 1) << 5 | op;
    FpOp o;
    V x;
    if (!scalar && sz && !q) return 0;
    memset(&o, 0, sizeof(o));
    o.p.conversion.is_unsigned = u;
    if (op >= 0x0C && op <= 0x0E && (size >> 1)) {
        if (op == 0x0C)
            o.p.compare.predicate = u ? SNX_IR_CMP_GE : SNX_IR_CMP_GT;
        else if (op == 0x0D)
            o.p.compare.predicate = u ? SNX_IR_CMP_LE : SNX_IR_CMP_EQ;
        else if (!u)
            o.p.compare.predicate = SNX_IR_CMP_LT;
        else
            return 0;
        o.kind = FP_LANES_COMPARE;
        fp_lanes(ir, &o, s, n, get_lanes(ir, vreg(ir, rn), s, n), LLVMConstNull(lanes_type(ir, s, n)), rd, q);
        return 1;
    }
    switch (k) {
        case 0x1A: /* FCVTNS, FCVTPS, FCVTMS, FCVTZS, FCVTAS and the unsigned ones: snx_fp_to_int's mode */
        case 0x3A:
        case 0x1B:
        case 0x3B:
        case 0x1C:
        case 0x5A:
        case 0x7A:
        case 0x5B:
        case 0x7B:
        case 0x5C:
            o.kind = FP_LANES_TO_INT;
            o.p.conversion.rounding =
                (k & 0x1F) == 0x1C ? 4 : (k & 0x1F) == 0x1A ? (size >> 1 ? 1 : 0) : (size >> 1 ? 3 : 2);
            break;
        case 0x1D: /* SCVTF, UCVTF */
        case 0x5D: o.kind = FP_LANES_FROM_INT; break;
        case 0x3D: /* FRECPE, FRSQRTE */
        case 0x7D:
            o.kind = FP_LANES_ESTIMATE;
            o.p.estimate.rsqrt = u;
            break;
        case 0x7F:
            if (scalar) return 0; /* FSQRT has no scalar AdvSIMD form */
            o.kind = FP_LANES_SQRT;
            break;
        default: return 0;
    }
    x = get_lanes(ir, vreg(ir, rn), s, n);
    fp_lanes(ir, &o, s, n, x, NULL, rd, q);
    return 1;
}

/* SCVTF, UCVTF, FCVTZS, FCVTZU (vector and scalar, fixed point): simd_shimm's conversions with fbits = 2*esize - imm */
static int fp_fixed(SnxIr* ir, SnxU32 w, unsigned scalar) {
    unsigned q = scalar ? 0 : (w >> 30) & 1, u = (w >> 29) & 1, immh = (w >> 19) & 15, immb = (w >> 16) & 7,
             op = (w >> 11) & 31, rn = (w >> 5) & 31, rd = w & 31, s, n;
    FpOp o;
    if ((op != 0x1C && op != 0x1F) || immh < 4) return 0; /* 32 and 64-bit lanes only */
    s = immh & 8 ? 3 : 2;
    if (s == 3 && !q && !scalar) return 0;
    n = scalar ? 1 : (q ? 128u : 64u) >> (3 + s);
    o.kind = op == 0x1C ? FP_LANES_FROM_INT : FP_LANES_TO_INT;
    o.p.conversion.rounding = 3; /* toward zero */
    o.p.conversion.is_unsigned = u;
    o.p.conversion.fractional_bits = 2 * (8u << s) - ((immh << 3) | immb);
    fp_lanes(ir, &o, s, n, get_lanes(ir, vreg(ir, rn), s, n), NULL, rd, q);
    return 1;
}

/* FCVTN, FCVTN2 (doubles to floats, floats to halves, FPCR's rounding) and FCVTXN, FCVTXN2 (doubles to floats,
   round to odd, also scalar): simd_2misc's snx_fp_convert lanes, into Vd's low half with the upper half zero, its
   upper half with the low half kept (the "2" forms), or for the scalar FCVTXN its low 32 bits */
static int fp_narrow(SnxIr* ir, SnxU32 w, unsigned scalar) {
    unsigned q = scalar ? 0 : (w >> 30) & 1, u = (w >> 29) & 1, size = (w >> 22) & 3, op = (w >> 12) & 31,
             rn = (w >> 5) & 31, rd = w & 31, from = size & 1 ? 64 : 32, s = size & 1 ? 3 : 2,
             n = scalar ? 1 : 128 / from, i;
    LLVMTypeRef t = lane_type(ir, s - 1);
    SnxFpPath path;
    SnxFpFast f;
    V x, r;
    if (op != 0x16 || (size >> 1) || (scalar && !u) || (u && from != 64)) return 0;
    x = get_lanes(ir, vreg(ir, rn), s, n);
    ir_fp_narrow_fast(ir, from, n, x, u, &f);
    path = snx_fp_path_begin(ir, f.ok);
    snx_fp_path_else(ir, &path, f.inexact);
    r = n == 1 ? NULL : LLVMGetUndef(LLVMVectorType(t, n));
    for (i = 0; i < n; ++i) {
        V lane =
            ir_cast(ir, ir_fp_narrow(ir, n == 1 ? x : LLVMBuildExtractElement(B, x, ir_i32(ir, i), ""), from, u), t, 0);
        r = n == 1 ? lane : LLVMBuildInsertElement(B, r, lane, ir_i32(ir, i), "");
    }
    write_narrow(ir, rd, snx_fp_path_merge(ir, &path, f.value, r), q);
    return 1;
}

/* ---- the integer and data-movement rest of snx_simd ---- */

/* DUP (element, scalar), the MOV Bd/Hd/Sd/Dd, Vn.T[i] alias: the element alone in Vd */
