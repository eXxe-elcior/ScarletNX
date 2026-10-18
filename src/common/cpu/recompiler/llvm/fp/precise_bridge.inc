/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
/* the helpers' prototypes in cpu.h: CPU* first, then u64 operands and int parameters */
static V fp_call(SnxIr* ir, const char* name, LLVMTypeRef ret, unsigned wide, V* operands, unsigned count) {
    LLVMTypeRef p[8];
    V args[8];
    unsigned i;
    p[0] = ir->t_ptr;
    args[0] = ir->c;
    for (i = 0; i < count; ++i) {
        p[i + 1] = i < wide ? ir->t_i64 : ir->t_i32;
        args[i + 1] = i < wide ? ir_u64(ir, operands[i]) : operands[i];
    }
    return ir_call(ir, ir_extern(ir, name, ret, p, count + 1), args, count + 1);
}
V ir_fp_binary(SnxIr* ir, V a, V b, unsigned width, unsigned op) {
    V x[4];
    x[0] = a, x[1] = b, x[2] = i32c(ir, width), x[3] = i32c(ir, op);
    return fp_call(ir, "snx_fp_binary", ir->t_i64, 2, x, 4);
}
V ir_fp_minmax(SnxIr* ir, V a, V b, unsigned width, unsigned minimum, unsigned numeric) {
    V x[5];
    x[0] = a, x[1] = b, x[2] = i32c(ir, width), x[3] = i32c(ir, minimum), x[4] = i32c(ir, numeric);
    return fp_call(ir, "snx_fp_minmax", ir->t_i64, 2, x, 5);
}
static V fma3(SnxIr* ir, V add, V a, V b, unsigned width) {
    V x[4];
    x[0] = add, x[1] = a, x[2] = b, x[3] = i32c(ir, width);
    return fp_call(ir, "snx_fp_fma", ir->t_i64, 3, x, 4);
}

/* One guarded arithmetic implementation for scalar FP and AdvSIMD FP.
 * All sources are SSA values captured before architectural writes, including
 * the FMA accumulator. This is translation-time dispatch on a decoded enum;
 * the generated guest code contains only the chosen operation and guard.
 */
V snx_fp_emit_arithmetic(SnxIr* ir, const SnxFpArithmetic* spec) {
    SnxFpFast f;
    SnxFpPath path;
    LLVMTypeRef lane_type = spec->width == 64 ? ir->t_i64 : ir->t_i32;
    V precise = NULL;
    unsigned i;
    ir_fp_fast(ir, (unsigned)spec->kind, spec->width, spec->lanes,
               spec->a, spec->b, spec->addend, &f);
    path = snx_fp_path_begin(ir, f.ok);
    snx_fp_path_else(ir, &path, f.inexact);
    for (i = 0; i < spec->lanes; ++i) {
        V a = spec->lanes == 1 ? spec->a : LLVMBuildExtractElement(B, spec->a, i32c(ir, i), "");
        V b = spec->b ? (spec->lanes == 1 ? spec->b
                         : LLVMBuildExtractElement(B, spec->b, i32c(ir, i), "")) : NULL;
        V result;
        /*
         * The vector fast guard is intentionally all-lanes.  If it fails,
         * re-check each lane individually on four-lane vectors.  The scalar
         * emitter retains the complete architectural guard and exact-helper
         * fallback, so a single NaN or subnormal need not send unrelated
         * ordinary lanes through software.  Maintain original lane order:
         * precise helpers may update cumulative FPSR state.
         *
         * For 1-2 lanes the branch/code-size overhead is unlikely to repay
         * the savings, so keep the existing software-only slow path.
         */
        if (spec->lanes >= SNX_FP_SELECTIVE_LANES_MIN) {
            V add = spec->addend ? LLVMBuildExtractElement(B, spec->addend, i32c(ir, i), "") : NULL;
            SnxFpArithmetic lane_spec = {spec->kind, spec->width, 1, a, b, add};
            result = snx_fp_emit_arithmetic(ir, &lane_spec);
        } else switch (spec->kind) {
            case SNX_FP_ARITH_ADD:
            case SNX_FP_ARITH_SUB:
            case SNX_FP_ARITH_MUL:
            case SNX_FP_ARITH_DIV:
                result = ir_fp_binary(ir, a, b, spec->width, (unsigned)spec->kind);
                break;
            case SNX_FP_ARITH_SQRT:
                result = ir_fp_unary(ir, a, spec->width, 0);
                break;
            case SNX_FP_ARITH_FMA: {
                V add = spec->lanes == 1 ? spec->addend
                        : LLVMBuildExtractElement(B, spec->addend, i32c(ir, i), "");
                result = fma3(ir, add, a, b, spec->width);
                break;
            }
            default:
                /* Decoders only construct one of the six arithmetic kinds. */
                return NULL;
        }
        result = ir_cast(ir, result, lane_type, 0);
        if (spec->lanes == 1)
            precise = result;
        else {
            if (i == 0) precise = LLVMGetUndef(LLVMTypeOf(f.value));
            precise = LLVMBuildInsertElement(B, precise, result, i32c(ir, i), "");
        }
    }
    return snx_fp_path_merge(ir, &path, f.value, precise);
}
V ir_fp_unary(SnxIr* ir, V bits, unsigned width, unsigned op) {
    V x[3];
    x[0] = bits, x[1] = i32c(ir, width), x[2] = i32c(ir, op);
    return fp_call(ir, "snx_fp_unary", ir->t_i64, 1, x, 3);
}
static V convert(SnxIr* ir, V bits, unsigned from, unsigned to, V rm) {
    V x[4];
    x[0] = bits, x[1] = i32c(ir, from), x[2] = i32c(ir, to), x[3] = rm;
    return fp_call(ir, "snx_fp_convert", ir->t_i64, 1, x, 4);
}
static V round_to(SnxIr* ir, V bits, unsigned width, V rm, unsigned exact) {
    V x[4];
    x[0] = bits, x[1] = i32c(ir, width), x[2] = rm, x[3] = i32c(ir, exact);
    return fp_call(ir, "snx_fp_round", ir->t_i64, 1, x, 4);
}
static V compare(SnxIr* ir, V a, V b, unsigned width, V signal) {
    V x[4];
    x[0] = a, x[1] = b, x[2] = i32c(ir, width), x[3] = signal;
    return fp_call(ir, "snx_fp_compare", ir->t_i32, 2, x, 4);
}
V ir_fp_to_int(SnxIr* ir, V bits, unsigned fp, unsigned integer, unsigned uns, unsigned fbits, unsigned rm) {
    V x[6];
    x[0] = bits, x[1] = i32c(ir, fp), x[2] = i32c(ir, integer), x[3] = i32c(ir, uns), x[4] = i32c(ir, fbits),
    x[5] = i32c(ir, rm);
    return fp_call(ir, "snx_fp_to_int", ir->t_i64, 1, x, 6);
}
V ir_fp_from_int(SnxIr* ir, V bits, unsigned integer, unsigned fp, unsigned uns, unsigned fbits) {
    V x[5];
    x[0] = bits, x[1] = i32c(ir, integer), x[2] = i32c(ir, fp), x[3] = i32c(ir, uns), x[4] = i32c(ir, fbits);
    return fp_call(ir, "snx_fp_from_int", ir->t_i64, 1, x, 5);
}
/* FPRM: (int)((c->fpcr >> 22) & 3) */
static V fprm(SnxIr* ir) {
    V fpcr = ir_cpu_load(ir, ir->t_i32, ir_off_fpcr());
    return LLVMBuildAnd(B, LLVMBuildLShr(B, fpcr, i32c(ir, 22), ""), i32c(ir, 3), "");
}
/* simd_2misc's FCVTN (FPRM) and FCVTXN (mode 5, round to odd) lane */
V ir_fp_narrow(SnxIr* ir, V bits, unsigned from, unsigned odd) {
    return convert(ir, bits, from, from / 2, odd ? i32c(ir, 5) : fprm(ir));
}

