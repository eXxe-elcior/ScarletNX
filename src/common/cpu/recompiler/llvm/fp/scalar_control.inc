/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
/* FCMP without a NaN, and without a denormal when FZ is set (flushing it sets IDC): the host comparison gives the
   flags, and no FPSR bit changes. Zeros compare as IEEE says, equal whatever their signs, with FZ or without */
static V compare_fast(SnxIr* ir, V a, V b, unsigned width, V* ok) {
    Shape s;
    V fa, fb, nzcv;
    shape(ir, width, 1, &s);
    a = ir_cast(ir, a, s.it, 0);
    b = ir_cast(ir, b, s.it, 0);
    fa = LLVMBuildBitCast(B, a, s.ft, "");
    fb = LLVMBuildBitCast(B, b, s.ft, "");
    *ok = no_nan_no_flush(ir, &s, a, b, fa, fb);
    nzcv = LLVMBuildSelect(B, LLVMBuildFCmp(B, LLVMRealOLT, fa, fb, ""), i32c(ir, 8), i32c(ir, 2), "");
    return LLVMBuildSelect(B, LLVMBuildFCmp(B, LLVMRealOEQ, fa, fb, ""), i32c(ir, 6), nzcv, "");
}

static void fcmp(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned width = index >> 2 ? 64 : 32;
    V b = index & 1 ? ir_i64(ir, 0) : operand(ir, F(16, 5), width), a = operand(ir, F(5, 5), width), ok, fast;
    LLVMBasicBlockRef slow, join;
    fast = compare_fast(ir, a, b, width, &ok);
    join = ir_fp_branch(ir, ok, &slow);
    set_nzcv(ir, fast);
    ir_fp_else(ir, join, slow, LLVMConstInt(ir->t_i1, 0, 0));
    set_nzcv(ir, compare(ir, a, b, width, i32c(ir, (index >> 1) & 1)));
    ir_fp_join(ir, join);
}

static void fccmp(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned width = index ? 64 : 32;
    LLVMBasicBlockRef taken = ir_block(ir, "fccmp"), flags = ir_block(ir, "nzcv"), done = ir_block(ir, "done");
    LLVMBuildCondBr(B, truth(ir, cond_ok(ir, F(12, 4))), taken, flags);
    LLVMPositionBuilderAtEnd(B, taken);
    {
        V a = operand(ir, F(5, 5), width), b = operand(ir, F(16, 5), width), ok, fast;
        LLVMBasicBlockRef slow, join;
        fast = compare_fast(ir, a, b, width, &ok);
        join = ir_fp_branch(ir, ok, &slow);
        set_nzcv(ir, fast);
        ir_fp_else(ir, join, slow, LLVMConstInt(ir->t_i1, 0, 0));
        set_nzcv(ir, compare(ir, a, b, width, i32c(ir, F(4, 1))));
        ir_fp_join(ir, join);
    }
    LLVMBuildBr(B, done);
    LLVMPositionBuilderAtEnd(B, flags);
    set_nzcv(ir, i32c(ir, F(0, 4)));
    LLVMBuildBr(B, done);
    LLVMPositionBuilderAtEnd(B, done);
}

static void fcsel(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned width = index ? 64 : 32;
    V ok = truth(ir, cond_ok(ir, F(12, 4)));
    set_bits(ir, F(0, 5), width, LLVMBuildSelect(B, ok, operand(ir, F(5, 5), width), operand(ir, F(16, 5), width), ""));
}

static void fmov_imm(SnxIr* ir, SnxU32 w, size_t index) {
    V imm = i32c(ir, F(13, 8));
    set_bits(ir, F(0, 5), index ? 64 : 32, ir_call_helper(ir, index ? SNX_IR_VFP_IMM64 : SNX_IR_VFP_IMM32, &imm, 1));
}

static void fp3(SnxIr* ir, SnxU32 w, size_t index) {
    unsigned o0 = index & 1, o1 = (index >> 1) & 1, width = index >> 2 ? 64 : 32;
    V a = operand(ir, F(10, 5), width), n = operand(ir, F(5, 5), width), m = operand(ir, F(16, 5), width);
    /* FMADD, FMSUB (-n), FNMADD (-a, -n), FNMSUB (-a) */
    SnxFpArithmetic spec;
    if (o1) a = sign(ir, a, width);
    if (o0 != o1) n = sign(ir, n, width);
    spec = (SnxFpArithmetic){SNX_FP_ARITH_FMA, width, 1, n, m, a};
    set_bits(ir, F(0, 5), width, snx_fp_emit_arithmetic(ir, &spec));
}

/* the host rounds to an integral value in one instruction (x86 SSE4.1 ROUNDSD, part of the CPUID sse4.2 gate;
   AArch64 FRINT*): llvm.floor and friends never become C library calls */
static int host_round(SnxIr* ir) {
    if (ir->features->x86) return ir->features->sse42;
    return !strncmp(LLVMGetTarget(ir->mod), "aarch64", 7);
}

/* 2^e, exactly (|e| <= 64 here) */
static double pow2(int e) {
    double r = 1.0;
    for (; e > 0; --e) r *= 2.0;
    for (; e < 0; ++e) r *= 0.5;
    return r;
}

int ir_fp_to_int_fast(SnxIr* ir, unsigned width, unsigned lanes, V bits, unsigned integer, unsigned uns, unsigned rm,
                      unsigned fbits, SnxFpFast* out) {
    static const char* const rounding[5] = {"llvm.roundeven", "llvm.ceil", "llvm.floor", "llvm.trunc", "llvm.round"};
    Shape s;
    V f, scaled, rounded, fz, limit, low, result, e, mantissa, ok;
    LLVMTypeRef it = LLVMIntTypeInContext(ir->ctx, integer);
    double top = integer == 64 ? 9223372036854775808.0 : 2147483648.0;
    if (rm != 3 && !host_round(ir)) return 0;
    shape(ir, width, lanes, &s);
    if (lanes > 1) it = LLVMVectorType(it, lanes);
    f = LLVMBuildBitCast(B, bits, s.ft, "");
    /* scaling by 2^fbits is exact, or overflows to an infinity, which is out of range */
    scaled = fbits ? LLVMBuildFMul(B, f, kfloat(ir, &s, pow2((int)fbits)), "") : f;
    rounded = rm == 3 ? scaled : fp_intrinsic(ir, rounding[rm], &s, &scaled, 1); /* FPToSI truncates by itself */
    e = exponent(ir, &s, bits);
    mantissa = LLVMBuildAnd(B, bits, kint(ir, &s, (((SnxU64)1 << s.mantissa) - 1)), "");
    fz = LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAnd(B, ir_cpu_load(ir, ir->t_i32, ir_off_fpcr()), i32c(ir, 1u << 24), ""),
                       i32c(ir, 0), "");
    /* finite, and no denormal to flush */
    ok = LLVMBuildAnd(B, all(ir, &s, LLVMBuildICmp(B, LLVMIntNE, e, kint(ir, &s, s.emax), "")),
                      LLVMBuildOr(B, LLVMBuildNot(B, fz, ""),
                                  all(ir, &s,
                                      LLVMBuildOr(B, LLVMBuildICmp(B, LLVMIntNE, e, kint(ir, &s, 0), ""),
                                                  LLVMBuildICmp(B, LLVMIntEQ, mantissa, kint(ir, &s, 0), ""), "")),
                                  ""),
                      "");
    if (uns) {
        limit = kfloat(ir, &s, top * 2);
        low = kfloat(ir, &s, 0.0); /* -0 compares equal: FCVTPU(-0.3) is 0, inexact */
        result = LLVMBuildFPToUI(B, rounded, it, "");
    } else {
        limit = kfloat(ir, &s, top);
        low = kfloat(ir, &s, -top);
        result = LLVMBuildFPToSI(B, rounded, it, "");
    }
    /* out of range the conversion is poison, used only on the slow path's side of the branch, which ignores it */
    out->ok = LLVMBuildAnd(B, ok,
                           all(ir, &s,
                               LLVMBuildAnd(B, LLVMBuildFCmp(B, LLVMRealOGE, rounded, low, ""),
                                            LLVMBuildFCmp(B, LLVMRealOLT, rounded, limit, ""), "")),
                           "");
    if (rm == 3) /* FPToSI rounded: the integer, converted back (exact while it is in range) */
        rounded = uns ? LLVMBuildUIToFP(B, result, s.ft, "") : LLVMBuildSIToFP(B, result, s.ft, "");
    out->inexact = any(ir, &s, LLVMBuildFCmp(B, LLVMRealUNE, rounded, scaled, ""));
    out->value = result;
    return 1;
}

void ir_fp_from_int_fast(SnxIr* ir, unsigned width, unsigned lanes, V v, unsigned uns, unsigned fbits, SnxFpFast* out) {
    Shape s;
    V f, x, small, p, p2;
    SnxU64 limit;
    shape(ir, width, lanes, &s);
    limit = (SnxU64)1 << s.precision;
    f = uns ? LLVMBuildUIToFP(B, v, s.ft, "") : LLVMBuildSIToFP(B, v, s.ft, "");
    /* scaling by 2^-fbits is exact: the smallest nonzero result, 2^-64, is a normal number */
    if (fbits) f = LLVMBuildFMul(B, f, kfloat(ir, &s, pow2(-(int)fbits)), "");
    /* exact for magnitudes up to 2^precision: one lane widened to 64 bits, lanes in their own width (as wide as the
       float, so 2^(precision+1) fits them) */
    if (lanes == 1) {
        x = uns ? ir_u64(ir, v) : ir_s64(ir, v);
        p = ir_i64(ir, limit);
        p2 = ir_i64(ir, 2 * limit);
    } else {
        x = v;
        p = kint(ir, &s, limit);
        p2 = kint(ir, &s, 2 * limit);
    }
    if (uns)
        small = LLVMBuildICmp(B, LLVMIntULE, x, p, "");
    else
        small = LLVMBuildICmp(B, LLVMIntULE, LLVMBuildAdd(B, x, p, ""), p2, "");
    out->ok = LLVMBuildAnd(B, rounds_to_nearest(ir), LLVMBuildOr(B, all(ir, &s, small), ixc_set(ir), ""), "");
    out->value = LLVMBuildBitCast(B, f, s.it, "");
    out->inexact = LLVMConstInt(ir->t_i1, 0, 0);
}

void ir_fp_estimate_fast(SnxIr* ir, unsigned rsqrt, unsigned width, unsigned lanes, V bits, SnxFpFast* out) {
    Shape s, n;
    V e, est, power, t;
    unsigned f, bias, k;
    shape(ir, width, lanes, &s);
    shape(ir, 32, lanes, &n); /* the estimate's arithmetic, in 32-bit lanes */
    f = s.mantissa;
    bias = s.emax >> 1;
    e = exponent(ir, &s, bits);
    if (!rsqrt) {
        /* recip_est(256 + i) = (2^19 / (513 + 2i) + 1) >> 1, i the fraction's top 8 bits. The quotient's floor comes
           from a float division: the divisor is odd and below 2^10, so the quotient is never an integer and lies
           more than 2^-10 from one, far beyond the division's rounding error (2^-15 below 1024) */
        V d;
        t = LLVMBuildAnd(B, LLVMBuildLShr(B, bits, kint(ir, &s, f - 8), ""), kint(ir, &s, 255), "");
        if (width == 64) t = LLVMBuildTrunc(B, t, n.it, "");
        d = LLVMBuildAdd(B, LLVMBuildShl(B, t, kint(ir, &n, 1), ""), kint(ir, &n, 513), "");
        est = LLVMBuildFPToUI(B, LLVMBuildFDiv(B, kfloat(ir, &n, 524288.0), LLVMBuildUIToFP(B, d, n.ft, ""), ""), n.it,
                              "");
        est = LLVMBuildLShr(B, LLVMBuildAdd(B, est, kint(ir, &n, 1), ""), kint(ir, &n, 1), "");
        /* exponents 1..emax-3: the reciprocal's, emax-2-e, is normal */
        power = LLVMBuildSub(B, kint(ir, &s, s.emax - 2), e, "");
        out->ok = exponent_in(ir, &s, bits, 1, s.emax - 3);
        power = LLVMBuildOr(B, LLVMBuildAnd(B, bits, kint(ir, &s, (SnxU64)1 << (width - 1)), ""),
                            LLVMBuildShl(B, power, kint(ir, &s, f), ""), "");
    } else {
        /* rsqrt_est's operand: 2*sc+1 for an odd exponent, sc = 128 + t (t the fraction's top 7 bits), and
           2*((sc&~1)+1) for an even one, sc = 256 + the top 8 bits: 257 + 2t and 514 + 4t. Then its binary search,
           nine halvings of [512, 1024) */
        V odd, a, lo = kint(ir, &n, 512), hi = kint(ir, &n, 1024);
        t = LLVMBuildAnd(B, LLVMBuildLShr(B, bits, kint(ir, &s, f - 7), ""), kint(ir, &s, 127), "");
        odd = LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAnd(B, e, kint(ir, &s, 1), ""), kint(ir, &s, 0), "");
        if (width == 64) t = LLVMBuildTrunc(B, t, n.it, "");
        a = LLVMBuildSelect(B, odd, LLVMBuildAdd(B, LLVMBuildShl(B, t, kint(ir, &n, 1), ""), kint(ir, &n, 257), ""),
                            LLVMBuildAdd(B, LLVMBuildShl(B, t, kint(ir, &n, 2), ""), kint(ir, &n, 514), ""), "");
        for (k = 0; k < 9; ++k) {
            V mid = LLVMBuildLShr(B, LLVMBuildAdd(B, lo, hi, ""), kint(ir, &n, 1), "");
            V below = LLVMBuildICmp(B, LLVMIntULT, LLVMBuildMul(B, LLVMBuildMul(B, a, mid, ""), mid, ""),
                                    kint(ir, &n, 1u << 28), "");
            lo = LLVMBuildSelect(B, below, mid, lo, "");
            hi = LLVMBuildSelect(B, below, hi, mid, "");
        }
        est = LLVMBuildLShr(B, LLVMBuildAdd(B, lo, kint(ir, &n, 1), ""), kint(ir, &n, 1), "");
        /* positive normal operands: the result exponent (3*bias - 1 - e) / 2 is normal */
        out->ok = LLVMBuildAnd(B, exponent_in(ir, &s, bits, 1, s.emax - 1),
                               all(ir, &s, LLVMBuildICmp(B, LLVMIntSGE, bits, kint(ir, &s, 0), "")), "");
        power =
            LLVMBuildShl(B, LLVMBuildLShr(B, LLVMBuildSub(B, kint(ir, &s, 3 * bias - 1), e, ""), kint(ir, &s, 1), ""),
                         kint(ir, &s, f), "");
    }
    est = LLVMBuildAnd(B, est, kint(ir, &n, 255), "");
    if (width == 64) est = LLVMBuildZExt(B, est, s.it, "");
    out->value = LLVMBuildOr(B, power, LLVMBuildShl(B, est, kint(ir, &s, f - 8), ""), "");
    out->inexact = LLVMConstInt(ir->t_i1, 0, 0);
}

void ir_fp_step_fast(SnxIr* ir, unsigned rsqrt, unsigned width, unsigned lanes, V a, V b, SnxFpFast* out) {
    Shape s;
    SnxU64 two;
    shape(ir, width, lanes, &s);
    two = (SnxU64)((s.emax >> 1) + 1) << s.mantissa;
    /* snx_fp_binary negates a and fuses: 2 + (-a)*b, or 3 + (-a)*b */
    ir_fp_fast(ir, SNX_IR_FP_FMA, width, lanes, LLVMBuildXor(B, a, kint(ir, &s, (SnxU64)1 << (width - 1)), ""), b,
               kint(ir, &s, two | (rsqrt ? (SnxU64)1 << (s.mantissa - 1) : 0)), out);
    if (rsqrt) {
        /* halving the rounded sum is the rounded half (FPMulAdd's scale) while the half stays out of the smallest
           binade, where it could be tiny before rounding */
        V half = LLVMBuildBitCast(
            B, LLVMBuildFMul(B, LLVMBuildBitCast(B, out->value, s.ft, ""), kfloat(ir, &s, 0.5), ""), s.it, "");
        out->ok = LLVMBuildAnd(B, out->ok, exponent_in(ir, &s, half, 2, s.emax - 1), "");
        out->value = half;
    }
}

/* snx_fp_convert's fp_pack for a normal operand whose exponent, rebiased, is a normal one of the narrow format: the
   magnitude shifted right by the difference of the fractions, less the difference of the biases in the exponent
   field, is the truncated result; the dropped bits round it (fp_quantize: to nearest even, or to odd), and a carry
   out of the fraction is the exponent's increment, as fp_pack's renormalization. Only round to nearest can carry
   into the infinity encoding, which takes the helper (OFC) */
void ir_fp_narrow_fast(SnxIr* ir, unsigned from, unsigned lanes, V bits, unsigned odd, SnxFpFast* out) {
    Shape s;
    unsigned to = from / 2, narrow_mantissa = from == 64 ? 23 : 10, narrow_emax = from == 64 ? 0xFF : 0x1F;
    unsigned shift, rebias;
    V sign, magnitude, truncated, dropped, lost, r, ok;
    LLVMTypeRef t = LLVMIntTypeInContext(ir->ctx, to);
    shape(ir, from, lanes, &s);
    shift = s.mantissa - narrow_mantissa;
    rebias = (s.emax >> 1) - (narrow_emax >> 1); /* 1023 - 127, 127 - 15 */
    sign = LLVMBuildAnd(B, bits, kint(ir, &s, (SnxU64)1 << (from - 1)), "");
    magnitude = LLVMBuildXor(B, bits, sign, "");
    ok = lane_exponent_in(ir, &s, bits, rebias + 1, rebias + narrow_emax - 1);
    truncated = LLVMBuildSub(B, LLVMBuildLShr(B, magnitude, kint(ir, &s, shift), ""),
                             kint(ir, &s, (SnxU64)rebias << narrow_mantissa), "");
    dropped = LLVMBuildAnd(B, magnitude, kint(ir, &s, ((SnxU64)1 << shift) - 1), "");
    lost = LLVMBuildICmp(B, LLVMIntNE, dropped, kint(ir, &s, 0), "");
    if (odd)
        r = LLVMBuildOr(B, truncated, LLVMBuildZExt(B, lost, s.it, ""), "");
    else {
        V half = kint(ir, &s, (SnxU64)1 << (shift - 1));
        V up = LLVMBuildOr(
            B, LLVMBuildICmp(B, LLVMIntUGT, dropped, half, ""),
            LLVMBuildAnd(B, LLVMBuildICmp(B, LLVMIntEQ, dropped, half, ""),
                         LLVMBuildTrunc(B, truncated, lanes > 1 ? LLVMVectorType(ir->t_i1, lanes) : ir->t_i1, ""), ""),
            "");
        r = LLVMBuildAdd(B, truncated, LLVMBuildZExt(B, up, s.it, ""), "");
        ok = LLVMBuildAnd(
            B, ok, LLVMBuildICmp(B, LLVMIntULT, r, kint(ir, &s, (SnxU64)narrow_emax << narrow_mantissa), ""), "");
    }
    r = LLVMBuildOr(B, r, LLVMBuildLShr(B, sign, kint(ir, &s, from - to), ""), "");
    out->value = LLVMBuildTrunc(B, r, lanes > 1 ? LLVMVectorType(t, lanes) : t, "");
    out->ok = all(ir, &s, ok);
    if (!odd) out->ok = LLVMBuildAnd(B, rounds_to_nearest(ir), out->ok, "");
    out->inexact = any(ir, &s, lost);
}

/* FCVT_INT and FCVT_FIX: index = opcode | rmode << 3 | ftype << 5 | sf << 7 */
static int fcvt(SnxIr* ir, SnxU32 w, size_t index, int fixed) {
    unsigned opcode = index & 7, rmode = (index >> 3) & 3, type = (index >> 5) & 3, sf = (unsigned)(index >> 7);
    unsigned rd = F(0, 5), rn = F(5, 5), integer = sf ? 64 : 32, width = type == 0 ? 32 : type == 1 ? 64 : 16;
    unsigned fbits = fixed ? 64 - F(10, 6) : 0;
    SnxFpFast f;
    switch (opcode) {
        case 0:
        case 1:
        case 4:
        case 5:
            if (type > 1) return 0;
            {
                unsigned rm = opcode >= 4 ? 4 : rmode;
                V a = operand(ir, rn, width);
                if (!ir_fp_to_int_fast(ir, width, 1, a, integer, opcode & 1, rm, fbits, &f)) {
                    ir_setx(ir, rd, ir_fp_to_int(ir, a, width, integer, opcode & 1, fbits, rm));
                    return 1;
                }
                {
                    SnxFpPath path = snx_fp_path_begin(ir, f.ok);
                    V fast_value = ir_u64(ir, f.value); /* preserve zero-extension for 32-bit output */
                    V precise_value;
                    snx_fp_path_else(ir, &path, f.inexact);
                    precise_value = ir_fp_to_int(ir, a, width, integer, opcode & 1, fbits, rm);
                    ir_setx(ir, rd, snx_fp_path_merge(ir, &path, fast_value, precise_value));
                }
            }
            return 1;
        case 2:
        case 3:
            if (type > 1) return 0;
            {
                V x = ir_rx(ir, rn);
                ir_fp_from_int_fast(ir, width, 1, LLVMBuildTrunc(B, x, LLVMIntTypeInContext(ir->ctx, integer), ""),
                                    opcode & 1, fbits, &f);
                {
                    SnxFpPath path = snx_fp_path_begin(ir, f.ok);
                    V precise_value;
                    snx_fp_path_else(ir, &path, f.inexact);
                    precise_value = ir_cast(ir, ir_fp_from_int(ir, x, integer, width, opcode & 1, fbits),
                                            width == 64 ? ir->t_i64 : ir->t_i32, 0);
                    set_bits(ir, rd, width, snx_fp_path_merge(ir, &path, f.value, precise_value));
                }
            }
            return 1;
        case 6: /* FMOV to a general register */
            if (type == 2)
                ir_setx(ir, rd, ir_vd(ir, rn, 1));
            else
                ir_setx(ir, rd, ir_u64(ir, operand(ir, rn, width)));
            return 1;
        case 7: /* FMOV from a general register */
            if (type == 2)
                ir_set_vd(ir, rd, 1, ir_rx(ir, rn));
            else
                set_bits(ir, rd, width, width == 64 ? ir_rx(ir, rn) : ir_u32(ir, ir_rx(ir, rn)));
            return 1;
    }
    return 0;
}

int snx_ir_fp_plain(SnxIr* ir, SnxU32 w, SnxU64 pc, const SnxA64Pattern* pattern, size_t index) {
    const char* name = pattern->name;
    (void)pc;
    if (!strcmp(name, "FP2")) return fp2(ir, w, index);
    if (!strcmp(name, "FP1")) return fp1(ir, w, index);
    if (!strcmp(name, "FCMP"))
        fcmp(ir, w, index);
    else if (!strcmp(name, "FCCMP"))
        fccmp(ir, w, index);
    else if (!strcmp(name, "FCSEL"))
        fcsel(ir, w, index);
    else if (!strcmp(name, "FMOV_IMM"))
        fmov_imm(ir, w, index);
    else if (!strcmp(name, "FP3"))
        fp3(ir, w, index);
    else if (!strcmp(name, "FCVT_INT"))
        return fcvt(ir, w, index, 0);
    else if (!strcmp(name, "FCVT_FIX"))
        return fcvt(ir, w, index, 1);
    else
        return 0;
    return 1;
}
