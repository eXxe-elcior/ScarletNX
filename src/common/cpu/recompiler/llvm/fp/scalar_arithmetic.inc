/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
/* SETSB/SETDB/SETH(d,e): t=(uN)(e); c->v[d].d[1]=0; c->v[d].d[0]=t; */
static void set_bits(SnxIr* ir, unsigned d, unsigned bits, V value) {
    V t = ir_cast(ir, value, LLVMIntTypeInContext(ir->ctx, bits), 0);
    ir_set_vd(ir, d, 1, ir_i64(ir, 0));
    ir_set_vd(ir, d, 0, ir_u64(ir, t));
}

/* c->v[n].s[0], .d[0] or .h[0] for ftype 0, 1, 3 */
static V operand(SnxIr* ir, unsigned n, unsigned width) {
    return ir_ve(ir, n, width == 64 ? 3 : width == 32 ? 2 : 1, 0);
}
static V sign(SnxIr* ir, V x, unsigned width) {
    return LLVMBuildXor(B, x, width == 64 ? ir_i64(ir, (SnxU64)1 << 63) : i32c(ir, 0x80000000UL), "");
}
static V truth(SnxIr* ir, V v) { return LLVMBuildICmp(B, LLVMIntNE, v, i32c(ir, 0), ""); }
static V cond_ok(SnxIr* ir, unsigned cond) {
    V args[2];
    args[0] = ir->c;
    args[1] = i32c(ir, cond);
    return ir_call_helper(ir, SNX_IR_COND_OK, args, 2);
}
static void set_nzcv(SnxIr* ir, V flags) {
    V args[2];
    args[0] = ir->c;
    args[1] = flags;
    ir_call_helper(ir, SNX_IR_SET_NZCV, args, 2);
}

static int fp2(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned op = index & 15, width = index >> 4 ? 64 : 32, rd = F(0, 5);
    V a = operand(ir, F(5, 5), width), b = operand(ir, F(16, 5), width), r;
    if (op < 4 || op == 8) {                            /* MUL, DIV, ADD, SUB, FNMUL: inline when exact */
        static const SnxFpArithmeticKind kinds[4] = {
            SNX_FP_ARITH_MUL, SNX_FP_ARITH_DIV, SNX_FP_ARITH_ADD, SNX_FP_ARITH_SUB
        };
        SnxFpArithmetic spec = {op == 8 ? SNX_FP_ARITH_MUL : kinds[op], width, 1, a, b, NULL};
        r = snx_fp_emit_arithmetic(ir, &spec);
        set_bits(ir, rd, width, op == 8 ? sign(ir, r, width) : r);
        return 1;
    }
    if (op < 8) { /* FMAX, FMIN, FMAXNM, FMINNM: inline without a NaN (or a denormal under FZ) */
        SnxFpPath path;
        SnxFpFast f;
        ir_fp_minmax_fast(ir, width, 1, a, b, op & 1, &f);
        path = snx_fp_path_begin(ir, f.ok);
        snx_fp_path_else(ir, &path, f.inexact);
        r = ir_cast(ir, ir_fp_minmax(ir, a, b, width, op & 1, (op >> 1) & 1),
                    width == 64 ? ir->t_i64 : ir->t_i32, 0);
        set_bits(ir, rd, width, snx_fp_path_merge(ir, &path, f.value, r));
        return 1;
    }
    return 0;
}

static int fp1(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned op = index & 63, type = (unsigned)(index >> 6), rd = F(0, 5), rn = F(5, 5);
    unsigned width = type == 0 ? 32 : type == 1 ? 64 : 16;
    V a = operand(ir, rn, width), r;
    if (type == 3) { /* FCVT from half */
        if (op != 4 && op != 5) return 0;
        set_bits(ir, rd, op == 4 ? 32 : 64, convert(ir, a, 16, op == 4 ? 32 : 64, fprm(ir)));
        return 1;
    }
    if (type > 1) return 0;
    switch (op) {
        case 0: r = a; break;
        case 1:
            r = LLVMBuildAnd(B, a, width == 64 ? ir_i64(ir, ~((SnxU64)1 << 63)) : i32c(ir, 0x7FFFFFFFUL), "");
            break;
        case 2: r = sign(ir, a, width); break;
        case 3: { /* SNX_FP_SQRT, inline when exact */
            SnxFpArithmetic spec = {SNX_FP_ARITH_SQRT, width, 1, a, NULL, NULL};
            set_bits(ir, rd, width, snx_fp_emit_arithmetic(ir, &spec));
            return 1;
        }
        case 4:
        case 5:
        case 7: {
            unsigned to = op == 7 ? 16 : op == 4 ? 32 : 64;
            if (to == width) return 0;
            if (to != 16) {
                /* single <-> double: widening is exact for normal values and zeros; narrowing takes the usual
                   bounds, inexact when widening back differs */
                Shape from, into;
                V f, r, ok, inexact;
                SnxFpPath path;
                V fast_bits, precise_bits;
                shape(ir, width, 1, &from);
                shape(ir, to, 1, &into);
                f = LLVMBuildBitCast(B, a, from.ft, "");
                if (to == 64) {
                    r = LLVMBuildFPExt(B, f, into.ft, "");
                    ok = LLVMBuildOr(
                        B, exponent_in(ir, &from, a, 1, from.emax - 1),
                        LLVMBuildICmp(B, LLVMIntEQ, LLVMBuildShl(B, a, kint(ir, &from, 1), ""), kint(ir, &from, 0), ""),
                        "");
                    inexact = LLVMConstInt(ir->t_i1, 0, 0);
                } else {
                    r = LLVMBuildFPTrunc(B, f, into.ft, "");
                    ok = LLVMBuildAnd(
                        B, LLVMBuildAnd(B, rounds_to_nearest(ir), exponent_in(ir, &from, a, 1, from.emax - 1), ""),
                        exponent_in(ir, &into, LLVMBuildBitCast(B, r, into.it, ""), 2, into.emax - 1), "");
                    inexact = LLVMBuildFCmp(B, LLVMRealUNE, LLVMBuildFPExt(B, r, from.ft, ""), f, "");
                }
                fast_bits = LLVMBuildBitCast(B, r, into.it, "");
                path = snx_fp_path_begin(ir, ok);
                snx_fp_path_else(ir, &path, inexact);
                precise_bits = ir_cast(ir, convert(ir, a, width, to, fprm(ir)), into.it, 0);
                set_bits(ir, rd, to, snx_fp_path_merge(ir, &path, fast_bits, precise_bits));
                return 1;
            }
            set_bits(ir, rd, to, convert(ir, a, width, to, fprm(ir)));
            return 1;
        }
        case 8:
        case 9:
        case 10:
        case 11:
        case 12: r = round_to(ir, a, width, i32c(ir, op - 8), 0); break;
        case 14: r = round_to(ir, a, width, fprm(ir), 1); break;
        case 15: r = round_to(ir, a, width, fprm(ir), 0); break;
        default: return 0;
    }
    set_bits(ir, rd, width, r);
    return 1;
}

/* a denormal: exponent 0 and a nonzero fraction (zero has exponent 0 too, and flushing it changes nothing) */
static V lane_denormal(SnxIr* ir, const Shape* s, V bits) {
    return LLVMBuildAnd(B, LLVMBuildNot(B, lane_exponent_in(ir, s, bits, 1, s->emax), ""),
                        LLVMBuildNot(B, lane_zero(ir, s, bits), ""), "");
}

/* neither operand a NaN in any lane, and FZ clear or no operand a denormal in any lane: fp_input then changes
   nothing and sets no IDC, and only a NaN can raise IOC */
static V no_nan_no_flush(SnxIr* ir, const Shape* s, V a, V b, V fa, V fb) {
    V fz =
        LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAnd(B, ir_cpu_load(ir, ir->t_i32, ir_off_fpcr()), i32c(ir, 1u << 24), ""),
                      i32c(ir, 0), "");
    V flush = LLVMBuildAnd(B, fz, any(ir, s, lane_or(ir, lane_denormal(ir, s, a), lane_denormal(ir, s, b))), "");
    return LLVMBuildNot(B, LLVMBuildOr(B, any(ir, s, LLVMBuildFCmp(B, LLVMRealUNO, fa, fb, "")), flush, ""), "");
}

void ir_fp_minmax_fast(SnxIr* ir, unsigned width, unsigned lanes, V a, V b, unsigned minimum, SnxFpFast* out) {
    Shape s;
    V fa, fb, gt, lt, tie;
    shape(ir, width, lanes, &s);
    fa = LLVMBuildBitCast(B, a, s.ft, "");
    fb = LLVMBuildBitCast(B, b, s.ft, "");
    gt = LLVMBuildFCmp(B, LLVMRealOGT, fa, fb, "");
    lt = LLVMBuildFCmp(B, LLVMRealOLT, fa, fb, "");
    /* neither larger: equal, so the same bits unless they are zeros of different signs */
    tie = minimum ? LLVMBuildOr(B, a, b, "") : LLVMBuildAnd(B, a, b, "");
    out->value = minimum ? LLVMBuildSelect(B, lt, a, LLVMBuildSelect(B, gt, b, tie, ""), "")
                         : LLVMBuildSelect(B, gt, a, LLVMBuildSelect(B, lt, b, tie, ""), "");
    out->ok = no_nan_no_flush(ir, &s, a, b, fa, fb);
    out->inexact = LLVMConstInt(ir->t_i1, 0, 0);
}

void ir_fp_compare_fast(SnxIr* ir, unsigned cmp, unsigned width, unsigned lanes, V a, V b, SnxFpFast* out) {
    static const LLVMRealPredicate predicates[5] = {LLVMRealOEQ, LLVMRealOGE, LLVMRealOGT, LLVMRealOLE, LLVMRealOLT};
    Shape s;
    V fa, fb;
    shape(ir, width, lanes, &s);
    fa = LLVMBuildBitCast(B, a, s.ft, "");
    fb = LLVMBuildBitCast(B, b, s.ft, "");
    out->value = LLVMBuildSExt(B, LLVMBuildFCmp(B, predicates[cmp], fa, fb, ""), s.it, "");
    out->ok = no_nan_no_flush(ir, &s, a, b, fa, fb);
    out->inexact = LLVMConstInt(ir->t_i1, 0, 0);
}

/* fcmpmask: snx_fp_compare's NZCV (8 less, 6 equal, 2 greater, 3 unordered) tested for the predicate */
V ir_fp_compare_mask(SnxIr* ir, unsigned cmp, V a, V b, unsigned width) {
    V flags = compare(ir, a, b, width, i32c(ir, cmp != SNX_IR_CMP_EQ)), hit;
    V less = LLVMBuildICmp(B, LLVMIntEQ, flags, i32c(ir, 8), "");
    V equal = LLVMBuildICmp(B, LLVMIntEQ, flags, i32c(ir, 6), "");
    V greater = LLVMBuildICmp(B, LLVMIntEQ, flags, i32c(ir, 2), "");
    switch (cmp) {
        case SNX_IR_CMP_EQ: hit = equal; break;
        case SNX_IR_CMP_GE: hit = LLVMBuildOr(B, equal, greater, ""); break;
        case SNX_IR_CMP_GT: hit = greater; break;
        case SNX_IR_CMP_LE: hit = LLVMBuildOr(B, less, equal, ""); break;
        default: hit = less; break;
    }
    return LLVMBuildSExt(B, hit, width == 64 ? ir->t_i64 : ir->t_i32, "");
}

