/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright 2026 the ScarletNX contributors
 *
 * Guarded floating-point results remain SSA values until the slow/native
 * paths join. The caller commits the destination once, after the merge.
 * These helpers run during IR generation: no runtime function-pointer dispatch.
 */
#ifndef SNX_IR_FP_PATHS_H
#define SNX_IR_FP_PATHS_H

#include "common/cpu/recompiler/llvm/ir.h"

/* Only apply lane-selective rechecking to larger vectors, where skipping
 * multiple software helper calls may repay additional cold-path code size.
 * Set to 5 to disable on the AArch64 NEON 2- and 4-lane FP formats.
 */
#ifndef SNX_FP_SELECTIVE_LANES_MIN
#define SNX_FP_SELECTIVE_LANES_MIN 4
#endif

/* Just like ir_fp_else(), but returns the ACTUAL fast predecessor of join.
 * Implemented in ir_fp.c; it handles optional FPSR-update blocks.
 */
LLVMBasicBlockRef ir_fp_else_capture(SnxIr* ir, LLVMBasicBlockRef join, LLVMBasicBlockRef slow, LLVMValueRef inexact);

typedef enum {
    SNX_FP_ARITH_ADD = SNX_IR_FP_ADD,
    SNX_FP_ARITH_SUB = SNX_IR_FP_SUB,
    SNX_FP_ARITH_MUL = SNX_IR_FP_MUL,
    SNX_FP_ARITH_DIV = SNX_IR_FP_DIV,
    SNX_FP_ARITH_SQRT = SNX_IR_FP_SQRT,
    SNX_FP_ARITH_FMA = SNX_IR_FP_FMA
} SnxFpArithmeticKind;

typedef struct {
    SnxFpArithmeticKind kind;
    unsigned width;      /* 32 or 64 bits */
    unsigned lanes;      /* scalar=1, or an AdvSIMD vector lane count */
    LLVMValueRef a;      /* integer bit pattern (scalar or LLVM vector) */
    LLVMValueRef b;      /* NULL for SQRT */
    LLVMValueRef addend; /* only for fused multiply-add */
} SnxFpArithmetic;

/* Return an SSA integer or integer vector; caller owns architectural writeback.
 * Native hardware is used only under ir_fp_fast's exactness/FPCR guard.
 * Fallback calls the existing precise snx_fp_* helper, lane by lane.
 */
LLVMValueRef snx_fp_emit_arithmetic(SnxIr* ir, const SnxFpArithmetic* spec);

typedef struct {
    LLVMBasicBlockRef slow;
    LLVMBasicBlockRef join;
    LLVMBasicBlockRef fast_exit;
} SnxFpPath;

static inline SnxFpPath snx_fp_path_begin(SnxIr* ir, LLVMValueRef ok) {
    SnxFpPath p;
    p.fast_exit = NULL;
    p.join = ir_fp_branch(ir, ok, &p.slow);
    return p;
}

static inline void snx_fp_path_else(SnxIr* ir, SnxFpPath* p, LLVMValueRef inexact) {
    p->fast_exit = ir_fp_else_capture(ir, p->join, p->slow, inexact);
}

/* Called with the builder in the precise fallback's final basic block.
 * Incoming types must match, including vector lane counts.
 */
static inline LLVMValueRef snx_fp_path_merge(SnxIr* ir, SnxFpPath* p, LLVMValueRef fast_value,
                                             LLVMValueRef slow_value) {
    LLVMBasicBlockRef slow_exit = LLVMGetInsertBlock(ir->b);
    LLVMValueRef phi;
    LLVMValueRef incoming[2] = {fast_value, slow_value};
    LLVMBasicBlockRef predecessors[2] = {p->fast_exit, slow_exit};
    ir_fp_join(ir, p->join);
    phi = LLVMBuildPhi(ir->b, LLVMTypeOf(fast_value), "fp.result");
    LLVMAddIncoming(phi, incoming, predecessors, 2);
    return phi;
}

#endif /* SNX_IR_FP_PATHS_H */
