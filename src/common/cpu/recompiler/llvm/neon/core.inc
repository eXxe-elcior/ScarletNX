/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Internal translation-unit section. Included once by the owning emitter. */
static LLVMTypeRef lane_type(SnxIr* ir, unsigned size) {
    return size == 0 ? ir->t_i8 : size == 1 ? ir->t_i16 : size == 2 ? ir->t_i32 : ir->t_i64;
}

/* lane index of 1<<size bytes in a V128 at pointer base (a register or a local) */
static V lane_pointer(SnxIr* ir, V base, unsigned size, unsigned index) {
    V offset = ir_i64(ir, (SnxU64)index << size);
    return LLVMBuildInBoundsGEP2(B, ir->t_i8, base, &offset, 1, "");
}
static V get(SnxIr* ir, V base, unsigned size, unsigned index) {
    return ir_load(ir, lane_type(ir, size), lane_pointer(ir, base, size, index));
}
static void set(SnxIr* ir, V base, unsigned size, unsigned index, V value) {
    ir_store(ir, ir_cast(ir, value, lane_type(ir, size), 0), lane_pointer(ir, base, size, index));
}
static V vreg(SnxIr* ir, unsigned n) { return ir_cpu(ir, ir_off_v(n)); }
/* V128 x = <source>: a local copy */
static void assign(SnxIr* ir, V destination, V source) {
    LLVMBuildMemCpy(B, destination, 1, source, 1, ir_i64(ir, 16));
}

/* lanes 0..n-1 of 1<<size bytes at base as one value: iN for one lane, <n x iN> otherwise */
static LLVMTypeRef lanes_type(SnxIr* ir, unsigned size, unsigned n) {
    return n == 1 ? lane_type(ir, size) : LLVMVectorType(lane_type(ir, size), n);
}
static V get_lanes(SnxIr* ir, V base, unsigned size, unsigned n) {
    V load = ir_load(ir, lanes_type(ir, size, n), base);
    LLVMSetAlignment(load, 1);
    return load;
}
static void set_lanes(SnxIr* ir, V base, V value) {
    V store = LLVMBuildStore(B, value, base);
    LLVMSetAlignment(store, 1);
}
static void write_lanes(SnxIr* ir, unsigned rd, V value, unsigned q);

static V broadcast(SnxIr* ir, V scalar, unsigned n) {
    V mask[16], v;
    unsigned i;
    if (n == 1) return scalar;
    /* NEON has at most sixteen 8-bit lanes.  One insert and a shuffle
       expose a genuine splat to the LLVM vectorizer, without an N-insert
       construction.  The shuffle indices are translation-time constants. */
    v = LLVMGetUndef(LLVMVectorType(LLVMTypeOf(scalar), n));
    v = LLVMBuildInsertElement(B, v, scalar, ir_i32(ir, 0), "");
    for (i = 0; i < n; ++i) mask[i] = ir_i32(ir, 0);
    return LLVMBuildShuffleVector(B, v, v, LLVMConstVector(mask, n), "");
}
static V negate_lanes(SnxIr* ir, V x, unsigned sz, unsigned n) {
    return LLVMBuildXor(B, x, broadcast(ir, sz ? ir_i64(ir, (SnxU64)1 << 63) : ir_i32(ir, 0x80000000UL), n), "");
}

int snx_ir_fast_simd(SnxIr* ir, SnxU32 w) {
    unsigned q, u, size, rd, rn, rm, op, hi, sz, lanes, scalar, l, h, mb, index, m, imm5, imm4, lane;
    q = (w >> 30) & 1;
    u = (w >> 29) & 1;
    size = (w >> 22) & 3;
    rd = w & 31;
    rn = (w >> 5) & 31;
    rm = (w >> 16) & 31;
    if ((w & 0x9F200400UL) == 0x0E200400UL) {
        op = (w >> 11) & 31;
        if (op == 3) {
            V a0, a1, b0, b1;
            if (!u && size == 2 && rn == rm) { /* MOV (ORR Vd,Vn,Vn) */
                ir_set_vd(ir, rd, 0, ir_vd(ir, rn, 0));
                ir_set_vd(ir, rd, 1, q ? ir_vd(ir, rn, 1) : ir_i64(ir, 0));
                return 1;
            }
            if (u && size) return 0;
            a0 = ir_vd(ir, rn, 0);
            a1 = ir_vd(ir, rn, 1);
            b0 = ir_vd(ir, rm, 0);
            b1 = ir_vd(ir, rm, 1);
            if (!u && (size & 1)) { /* &~, |~ */
                b0 = LLVMBuildNot(B, b0, "");
                b1 = LLVMBuildNot(B, b1, "");
            }
            if (u) {
                ir_set_vd(ir, rd, 0, LLVMBuildXor(B, a0, b0, ""));
                ir_set_vd(ir, rd, 1, q ? LLVMBuildXor(B, a1, b1, "") : ir_i64(ir, 0));
            } else if (size < 2) {
                ir_set_vd(ir, rd, 0, LLVMBuildAnd(B, a0, b0, ""));
                ir_set_vd(ir, rd, 1, q ? LLVMBuildAnd(B, a1, b1, "") : ir_i64(ir, 0));
            } else {
                ir_set_vd(ir, rd, 0, LLVMBuildOr(B, a0, b0, ""));
                ir_set_vd(ir, rd, 1, q ? LLVMBuildOr(B, a1, b1, "") : ir_i64(ir, 0));
            }
            return 1;
        }
        if (op >= 0x18) { /* NEON floating arithmetic, including FMLA/FMLS */
            SnxFpArithmetic spec;
            SnxFpArithmeticKind kind = SNX_FP_ARITH_ADD;
            unsigned acc = 0, negated = 0;
            V a, result;
            sz = size & 1;
            hi = size >> 1;
            if (sz && !q) return 0;
            lanes = sz ? 2 : (q ? 4 : 2);
            if (!u && !hi && op == 0x1A) kind = SNX_FP_ARITH_ADD;
            else if (!u && hi && op == 0x1A) kind = SNX_FP_ARITH_SUB;
            else if (u && !hi && op == 0x1B) kind = SNX_FP_ARITH_MUL;
            else if (u && !hi && op == 0x1F) kind = SNX_FP_ARITH_DIV;
            else if (!u && op == 0x19) {
                kind = SNX_FP_ARITH_FMA;
                acc = 1;
                negated = hi;
            } else return 0;
            a = get_lanes(ir, vreg(ir, rn), sz ? 3 : 2, lanes);
            if (negated) a = negate_lanes(ir, a, sz, lanes);
            spec = (SnxFpArithmetic){kind, sz ? 64 : 32, lanes,
                a, get_lanes(ir, vreg(ir, rm), sz ? 3 : 2, lanes),
                acc ? get_lanes(ir, vreg(ir, rd), sz ? 3 : 2, lanes) : NULL};
            result = snx_fp_emit_arithmetic(ir, &spec);
            write_lanes(ir, rd, result, q);
            return 1;
        }
        if (op == 0x10 && (q || size < 3)) { /* ADD, SUB (vector): one SSA vector operation */
            V a, b, r;
            lanes = (q ? 16 : 8) >> size;
            a = get_lanes(ir, vreg(ir, rn), size, lanes);
            b = get_lanes(ir, vreg(ir, rm), size, lanes);
            r = u ? LLVMBuildSub(B, a, b, "") : LLVMBuildAdd(B, a, b, "");
            /* Q=0 writes the low 64 bits and clears the upper 64. */
            write_lanes(ir, rd, r, q);
            return 1;
        }
        return 0;
    }
    if ((w & 0x9F000400UL) == 0x0F000000UL || (w & 0xDF000400UL) == 0x5F000000UL) {
        /* FMLA/FMLS/FMUL by element. Snapshot all operands before writeback. */
        SnxFpArithmetic spec;
        V a, e, vec_b, result;
        unsigned s;
        scalar = (w >> 28) & 1;
        op = (w >> 12) & 15;
        if (u || (op != 1 && op != 5 && op != 9) || size < 2) return 0;
        sz = size & 1;
        l = (w >> 21) & 1;
        h = (w >> 11) & 1;
        mb = (w >> 20) & 1;
        if (sz && l) return 0;
        index = sz ? h : ((h << 1) | l);
        m = (mb << 4) | ((w >> 16) & 15);
        if (!scalar && sz && !q) return 0;
        lanes = scalar ? 1 : (sz ? 2 : (q ? 4 : 2));
        s = sz ? 3 : 2;
        a = get_lanes(ir, vreg(ir, rn), s, lanes);
        if (op == 5) a = negate_lanes(ir, a, sz, lanes);
        e = get(ir, vreg(ir, m), s, index);
        vec_b = broadcast(ir, e, lanes);
        spec = (SnxFpArithmetic){op == 9 ? SNX_FP_ARITH_MUL : SNX_FP_ARITH_FMA,
            sz ? 64 : 32, lanes, a, vec_b,
            op == 9 ? NULL : get_lanes(ir, vreg(ir, rd), s, lanes)};
        result = snx_fp_emit_arithmetic(ir, &spec);
        set_lanes(ir, vreg(ir, rd), result);
        if (scalar || !q) {
            if (scalar && !sz) {
                /* Scalar S writes zero to the entire remainder of Vd. */
                set(ir, vreg(ir, rd), 2, 1, ir_i32(ir, 0));
            }
            ir_set_vd(ir, rd, 1, ir_i64(ir, 0));
        }
        return 1;
    }
    if ((w & 0x9FE08400UL) == 0x0E000400UL) { /* DUP, INS, UMOV */
        imm5 = (w >> 16) & 31;
        imm4 = (w >> 11) & 15;
        if (!imm5) return 0;
        sz = 0;
        while (!(imm5 & (1U << sz))) ++sz;
        if (sz > 3) return 0;
        index = imm5 >> (sz + 1);
        lane = sz;
        lanes = (q ? 16 : 8) >> sz;
        if (!u && imm4 == 7 && ((sz == 3) == !!q)) {
            if (rd != 31) ir_set_x(ir, rd, ir_u64(ir, ir_ve(ir, rn, lane, index)));
            return 1;
        }
        if (!u && imm4 == 3 && q) {
            ir_set_ve(ir, rd, lane, index, rn == 31 ? ir_i64(ir, 0) : ir_x(ir, rn));
            return 1;
        }
        if (u && q) {
            ir_set_ve(ir, rd, lane, index, ir_ve(ir, rn, lane, imm4 >> sz));
            return 1;
        }
        if (!u && (imm4 == 0 || imm4 == 1) && !(sz == 3 && !q)) {
            V e;
            if (!imm4)
                e = ir_ve(ir, rn, lane, index);
            else
                e = ir_cast(ir, rn == 31 ? ir_i64(ir, 0) : ir_x(ir, rn), lane_type(ir, lane), 0);
            write_lanes(ir, rd, broadcast(ir, e, lanes), q);
            return 1;
        }
        return 0;
    }
    if ((w & 0xFFFFFFE0UL) == 0x2F00E400UL || (w & 0xFFFFFFE0UL) == 0x6F00E400UL) { /* MOVI Vd, #0 */
        ir_set_vd(ir, rd, 0, ir_i64(ir, 0));
        ir_set_vd(ir, rd, 1, ir_i64(ir, 0));
        return 1;
    }
    return 0;
}

/* SIMD_LDST, SIMD_LDST1, SIMD_DP, SIMD_DP2: if(!snx_simd(c,M,w)) c->rt->undef(c,pc,w); */
int snx_ir_simd_plain(SnxIr* ir, SnxU32 word, SnxU64 pc, const SnxA64Pattern* pattern, size_t index) {
    LLVMTypeRef p[3];
    V args[3], handled;
    LLVMBasicBlockRef undefined, done;
    (void)index;
    if (strcmp(pattern->name, "SIMD_LDST") && strcmp(pattern->name, "SIMD_LDST1") && strcmp(pattern->name, "SIMD_DP") &&
        strcmp(pattern->name, "SIMD_DP2"))
        return 0;
    p[0] = ir->t_ptr;
    p[1] = ir->t_ptr;
    p[2] = ir->t_i32;
    args[0] = ir->c;
    args[1] = ir->M;
    args[2] = ir_i32(ir, word);
    handled = ir_call(ir, ir_extern(ir, "snx_simd", ir->t_i32, p, 3), args, 3);
    undefined = ir_block(ir, "simd_undef");
    done = ir_block(ir, "simd_done");
    ir_unlikely(ir, LLVMBuildCondBr(B, LLVMBuildICmp(B, LLVMIntEQ, handled, ir_i32(ir, 0), ""), undefined, done));
    LLVMPositionBuilderAtEnd(B, undefined);
    ir_undef(ir, pc, word);
    LLVMBuildBr(B, done);
    LLVMPositionBuilderAtEnd(B, done);
    return 1;
}

