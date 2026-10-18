/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
static V i32c(SnxIr* ir, SnxU32 v) { return ir_i32(ir, v); }

/* ---- fast paths ---- */

typedef struct {
    LLVMTypeRef it, ft; /* lane types: iN and its float */
    unsigned lanes, mantissa, emax, precision;
} Shape;

static void shape(SnxIr* ir, unsigned width, unsigned lanes, Shape* s) {
    LLVMTypeRef i = width == 64 ? ir->t_i64 : ir->t_i32;
    LLVMTypeRef f = width == 64 ? LLVMDoubleTypeInContext(ir->ctx) : LLVMFloatTypeInContext(ir->ctx);
    s->lanes = lanes;
    s->it = lanes > 1 ? LLVMVectorType(i, lanes) : i;
    s->ft = lanes > 1 ? LLVMVectorType(f, lanes) : f;
    s->mantissa = width == 64 ? 52 : 23;
    s->emax = width == 64 ? 0x7FF : 0xFF;
    s->precision = s->mantissa + 1;
}
static V splat(const Shape* s, V scalar) {
    V values[4];
    unsigned i;
    if (s->lanes == 1) return scalar;
    for (i = 0; i < s->lanes; ++i) values[i] = scalar;
    return LLVMConstVector(values, s->lanes);
}
static V kint(SnxIr* ir, const Shape* s, SnxU64 v) {
    (void)ir;
    return splat(s, LLVMConstInt(s->lanes > 1 ? LLVMGetElementType(s->it) : s->it, v, 0));
}
static V kfloat(SnxIr* ir, const Shape* s, double v) {
    (void)ir;
    return splat(s, LLVMConstReal(s->lanes > 1 ? LLVMGetElementType(s->ft) : s->ft, v));
}
/* every lane / some lane of an i1 (vector) */
static V all(SnxIr* ir, const Shape* s, V c) {
    return s->lanes > 1 ? ir_intrinsic(ir, "llvm.vector.reduce.and", LLVMTypeOf(c), &c, 1) : c;
}
static V any(SnxIr* ir, const Shape* s, V c) {
    return s->lanes > 1 ? ir_intrinsic(ir, "llvm.vector.reduce.or", LLVMTypeOf(c), &c, 1) : c;
}
static V exponent(SnxIr* ir, const Shape* s, V bits) {
    return LLVMBuildAnd(B, LLVMBuildLShr(B, bits, kint(ir, s, s->mantissa), ""), kint(ir, s, s->emax), "");
}
/* lo <= exponent <= hi in each lane (i1, or a vector of them) */
static V lane_exponent_in(SnxIr* ir, const Shape* s, V bits, unsigned lo, unsigned hi) {
    return LLVMBuildICmp(B, LLVMIntULE, LLVMBuildSub(B, exponent(ir, s, bits), kint(ir, s, lo), ""),
                         kint(ir, s, hi - lo), "");
}
/* every lane */
static V exponent_in(SnxIr* ir, const Shape* s, V bits, unsigned lo, unsigned hi) {
    return all(ir, s, lane_exponent_in(ir, s, bits, lo, hi));
}
static V rounds_to_nearest(SnxIr* ir) {
    V fpcr = ir_cpu_load(ir, ir->t_i32, ir_off_fpcr());
    /* Conservative: do not use native arithmetic when any ARM FP
     * exception trap is enabled.  Sticky FPSR bits do not suppress
     * architecturally enabled traps on subsequently raised exceptions.
     * FPCR bits IOE..IXE and IDE are model-defined trap enables; the
     * software helper remains responsible for those configurations. */
    const SnxU32 trap_enables = (0x1Fu << 8) | (1u << 15);
    return LLVMBuildICmp(B, LLVMIntEQ,
        LLVMBuildAnd(B, fpcr, i32c(ir, (3u << 22) | trap_enables), ""), i32c(ir, 0), "");
}
static V ixc_set(SnxIr* ir) {
    V fpsr = ir_cpu_load(ir, ir->t_i32, ir_off_fpcr() + 4);
    return LLVMBuildICmp(B, LLVMIntNE, LLVMBuildAnd(B, fpsr, i32c(ir, 1u << 4), ""), i32c(ir, 0), "");
}
/* the host fuses multiply-add in hardware (x86 FMA3 from CPUID; AArch64 always): llvm.fma is then one instruction,
   never a call into a C library the module doesn't link */
static int host_fma(SnxIr* ir) {
    if (ir->features->x86) return ir->features->fma;
    return !strncmp(LLVMGetTarget(ir->mod), "aarch64", 7);
}
static V fp_intrinsic(SnxIr* ir, const char* name, const Shape* s, V* args, unsigned count) {
    return ir_intrinsic(ir, name, s->ft, args, count);
}
static V nonzero(SnxIr* ir, const Shape* s, V f) {
    return any(ir, s, LLVMBuildFCmp(B, LLVMRealUNE, f, kfloat(ir, s, 0.0), ""));
}

/* +0 or -0 in each lane */
static V lane_zero(SnxIr* ir, const Shape* s, V bits) {
    SnxU64 magnitude = ((SnxU64)s->emax << s->mantissa) | (((SnxU64)1 << s->mantissa) - 1);
    return LLVMBuildICmp(B, LLVMIntEQ, LLVMBuildAnd(B, bits, kint(ir, s, magnitude), ""), kint(ir, s, 0), "");
}
static V lane_or(SnxIr* ir, V x, V y) { return LLVMBuildOr(B, x, y, ""); }
static V lane_and(SnxIr* ir, V x, V y) { return LLVMBuildAnd(B, x, y, ""); }

/* IXC is sticky. Once set, determining the exact rounding residual cannot
 * change FPSR, provided the main guard excludes ALL other exceptional cases.
 * Branch around TwoSum / FMA residual arithmetic at runtime rather than using
 * select(): select would evaluate the expensive expression unconditionally.
 * The FPCR/result/input guards in ir_fp_fast still apply to both branches.
 */
static V residual_if_ixc_clear(SnxIr* ir, const Shape* s, unsigned op, V fa, V fb, V r, V ixc) {
    LLVMBasicBlockRef check = ir_block(ir, "fp_check_ixc");
    LLVMBasicBlockRef skip = ir_block(ir, "fp_ixc_sticky");
    LLVMBasicBlockRef join = ir_block(ir, "fp_ixc_join");
    LLVMBasicBlockRef check_exit, skip_exit;
    V precise, phi, inputs[2];
    LLVMBasicBlockRef blocks[2];
    LLVMBuildCondBr(B, ixc, skip, check);
    LLVMPositionBuilderAtEnd(B, check);
    if (op == SNX_IR_FP_ADD || op == SNX_IR_FP_SUB) {
        /* TwoSum: exact rounding error for normal IEEE round-to-nearest. */
        V y = op == SNX_IR_FP_SUB ? LLVMBuildFNeg(B, fb, "") : fb;
        V bp = LLVMBuildFSub(B, r, fa, "");
        V ap = LLVMBuildFSub(B, r, bp, "");
        precise = nonzero(ir, s, LLVMBuildFAdd(B,
            LLVMBuildFSub(B, fa, ap, ""), LLVMBuildFSub(B, y, bp, ""), ""));
    } else {
        V args[3];
        if (op == SNX_IR_FP_MUL) {
            args[0] = fa;
            args[1] = fb;
            args[2] = LLVMBuildFNeg(B, r, "");
        } else if (op == SNX_IR_FP_DIV) {
            args[0] = LLVMBuildFNeg(B, r, "");
            args[1] = fb;
            args[2] = fa;
        } else { /* SQRT */
            args[0] = LLVMBuildFNeg(B, r, "");
            args[1] = r;
            args[2] = fa;
        }
        precise = nonzero(ir, s, fp_intrinsic(ir, "llvm.fma", s, args, 3));
    }
    check_exit = LLVMGetInsertBlock(B);
    LLVMBuildBr(B, join);
    LLVMPositionBuilderAtEnd(B, skip);
    skip_exit = LLVMGetInsertBlock(B);
    LLVMBuildBr(B, join);
    LLVMPositionBuilderAtEnd(B, join);
    inputs[0] = precise;
    inputs[1] = LLVMConstInt(ir->t_i1, 0, 0);
    blocks[0] = check_exit;
    blocks[1] = skip_exit;
    phi = LLVMBuildPhi(B, ir->t_i1, "fp.new_ixc");
    LLVMAddIncoming(phi, inputs, blocks, 2);
    return phi;
}

void ir_fp_fast(SnxIr* ir, unsigned op, unsigned width, unsigned lanes, V a, V b, V add, SnxFpFast* out) {
    Shape s;
    V fa, fb, r, lane_in, zero_ok, exact_ok, args[3], ixc, za, zb = NULL, result;
    unsigned p, top;
    shape(ir, width, lanes, &s);
    p = s.precision;
    top = s.emax - 1;
    fa = LLVMBuildBitCast(B, a, s.ft, "");
    fb = b ? LLVMBuildBitCast(B, b, s.ft, "") : NULL;
    /* operands normal or zero: zeros follow IEEE's rules on the host, which are the architecture's (FPAdd, FPMul,
       FPMulAdd, FPDiv and FPSqrt give the same zero and sign in round to nearest, and raise nothing for them) */
    za = lane_zero(ir, &s, a);
    lane_in = lane_or(ir, lane_exponent_in(ir, &s, a, 1, top), za);
    if (b) {
        zb = lane_zero(ir, &s, b);
        lane_in = lane_and(ir, lane_in, lane_or(ir, lane_exponent_in(ir, &s, b, 1, top), zb));
    }
    /* a zero result is exact when it comes from a zero operand. Nonzero operands whose result rounds to zero
       underflowed (UFC, IXC), except for a sum: two floats that cancel do so exactly */
    zero_ok = splat(&s, LLVMConstInt(ir->t_i1, 1, 0));
    ixc = ixc_set(ir);
    exact_ok = ixc;
    out->inexact = LLVMConstInt(ir->t_i1, 0, 0);
    switch (op) {
        case SNX_IR_FP_ADD:
        case SNX_IR_FP_SUB: {
            V y = op == SNX_IR_FP_SUB ? LLVMBuildFNeg(B, fb, "") : fb;
            r = LLVMBuildFAdd(B, fa, y, "");
            out->inexact = residual_if_ixc_clear(ir, &s, op, fa, fb, r, ixc);
            exact_ok = LLVMConstInt(ir->t_i1, 1, 0);
            break;
        }
        case SNX_IR_FP_MUL:
            zero_ok = lane_or(ir, za, zb);
            r = LLVMBuildFMul(B, fa, fb, "");
            if (host_fma(ir)) { /* a*b - r, exact while the result is at least 2^(emin+p) or a zero operand's */
                out->inexact = residual_if_ixc_clear(ir, &s, op, fa, fb, r, ixc);
                exact_ok = LLVMBuildOr(
                    B, ixc,
                    all(ir, &s,
                        lane_or(ir, lane_exponent_in(ir, &s, LLVMBuildBitCast(B, r, s.it, ""), p + 2, top - 1),
                                zero_ok)),
                    "");
            }
            break;
        case SNX_IR_FP_DIV:
            lane_in = lane_and(ir, lane_in, LLVMBuildNot(B, zb, "")); /* by zero: DZC, or IOC for 0/0 */
            zero_ok = za;
            r = LLVMBuildFDiv(B, fa, fb, "");
            if (host_fma(ir)) { /* a - r*b, exact under the same bounds on a and r, or for a zero dividend */
                out->inexact = residual_if_ixc_clear(ir, &s, op, fa, fb, r, ixc);
                exact_ok = LLVMBuildOr(
                    B, ixc,
                    all(ir, &s,
                        lane_or(ir,
                                lane_and(ir, lane_exponent_in(ir, &s, LLVMBuildBitCast(B, r, s.it, ""), p + 2, top - 1),
                                         lane_exponent_in(ir, &s, a, p + 2, top - 1)),
                                za)),
                    "");
            }
            break;
        case SNX_IR_FP_SQRT: {
            /* sqrt(-0) is -0 without a flag; other negative operands are invalid */
            lane_in = lane_and(ir, lane_in, lane_or(ir, LLVMBuildICmp(B, LLVMIntSGE, a, kint(ir, &s, 0), ""), za));
            zero_ok = za;
            r = fp_intrinsic(ir, "llvm.sqrt", &s, &fa, 1);
            if (host_fma(ir)) { /* a - r*r */
                out->inexact = residual_if_ixc_clear(ir, &s, op, fa, fb, r, ixc);
                exact_ok =
                    LLVMBuildOr(B, ixc, all(ir, &s, lane_or(ir, lane_exponent_in(ir, &s, a, p + 2, top - 1), za)), "");
            }
            break;
        }
        default: { /* FMA: add + a*b, fused; whether it is exact isn't cheap to know, so only once IXC is set, or when
                      every product is a zero operand's (the result is then add, or a zero by IEEE's sign rules) */
            V zp = lane_or(ir, za, zb);
            args[0] = fa, args[1] = fb, args[2] = LLVMBuildBitCast(B, add, s.ft, "");
            lane_in = lane_and(ir, lane_in, lane_or(ir, lane_exponent_in(ir, &s, add, 1, top), lane_zero(ir, &s, add)));
            zero_ok = zp;
            r = host_fma(ir) ? fp_intrinsic(ir, "llvm.fma", &s, args, 3) : NULL;
            if (r)
                exact_ok = LLVMBuildOr(B, ixc, all(ir, &s, zp), "");
            else { /* no hardware FMA: never fast */
                r = LLVMBuildFAdd(B, args[2], LLVMBuildFMul(B, fa, fb, ""), "");
                exact_ok = LLVMConstInt(ir->t_i1, 0, 0);
            }
            break;
        }
    }
    out->value = LLVMBuildBitCast(B, r, s.it, "");
    /* no denormal or smallest-binade result (tininess is detected before rounding), no overflow, and a zero only
       where zero_ok says it is exact */
    result =
        lane_or(ir, lane_exponent_in(ir, &s, out->value, 2, top), lane_and(ir, lane_zero(ir, &s, out->value), zero_ok));
    out->ok = LLVMBuildAnd(B, LLVMBuildAnd(B, rounds_to_nearest(ir), all(ir, &s, lane_and(ir, lane_in, result)), ""),
                           exact_ok, "");
}

LLVMBasicBlockRef ir_fp_branch(SnxIr* ir, V ok, LLVMBasicBlockRef* slow) {
    LLVMBasicBlockRef fast = ir_block(ir, "fp_fast"), join = ir_block(ir, "fp_join");
    LLVMMetadataRef md[3];
    V br;
    *slow = ir_block(ir, "fp_slow");
    br = LLVMBuildCondBr(B, ok, fast, *slow);
    md[0] = LLVMMDStringInContext2(ir->ctx, "branch_weights", 14);
    md[1] = LLVMValueAsMetadata(ir_i32(ir, 2000));
    md[2] = LLVMValueAsMetadata(ir_i32(ir, 1));
    LLVMSetMetadata(br, LLVMGetMDKindIDInContext(ir->ctx, "prof", 4),
                    LLVMMetadataAsValue(ir->ctx, LLVMMDNodeInContext2(ir->ctx, md, 3)));
    LLVMPositionBuilderAtEnd(B, fast);
    return join;
}
/* Fast-path IXC is the only FPSR bit updated here. A sticky IXC needs no
 * write, and a false inexact condition must not store unchanged CPU state.
 * Keep ir_fp_else()'s public interface for the other emitters. */
LLVMBasicBlockRef ir_fp_else_capture(SnxIr* ir, LLVMBasicBlockRef join, LLVMBasicBlockRef slow, V inexact) {
    if (!LLVMIsConstant(inexact) || LLVMConstIntGetZExtValue(inexact)) {
        LLVMBasicBlockRef update = ir_block(ir, "fp_set_ixc");
        LLVMBasicBlockRef done = ir_block(ir, "fp_ixc_done");
        size_t off = ir_off_fpcr() + 4;
        LLVMBuildCondBr(B, inexact, update, done);
        LLVMPositionBuilderAtEnd(B, update);
        {
            V fpsr = ir_cpu_load(ir, ir->t_i32, off);
            ir_cpu_store(ir, LLVMBuildOr(B, fpsr, i32c(ir, 1u << 4), ""), off);
        }
        LLVMBuildBr(B, done);
        LLVMPositionBuilderAtEnd(B, done);
    }
    {
        LLVMBasicBlockRef fast_exit = LLVMGetInsertBlock(B);
        LLVMBuildBr(B, join);
        LLVMPositionBuilderAtEnd(B, slow);
        return fast_exit;
    }
}
void ir_fp_else(SnxIr* ir, LLVMBasicBlockRef join, LLVMBasicBlockRef slow, V inexact) {
    (void)ir_fp_else_capture(ir, join, slow, inexact);
}
void ir_fp_join(SnxIr* ir, LLVMBasicBlockRef join) {
    LLVMBuildBr(B, join);
    LLVMPositionBuilderAtEnd(B, join);
}

