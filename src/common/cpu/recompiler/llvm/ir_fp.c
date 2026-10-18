/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors
   Ported in part from QEMU (target/arm/tcg/translate-a64.c, helper-a64.c, vec_helper.c, vfp_helper.c,
   crypto_helper.c), LGPL-2.1-or-later:
   Copyright (c) 2013 Alexander Graf <agraf@suse.de>
   Copyright (c) 2018 Linaro
   Copyright (C) 2013 - 2018 Linaro Ltd <ard.biesheuvel@linaro.org>
   Copyright (c) 2003 Fabrice Bellard */
/* Scalar floating point: the FP patterns of a64_templates.inc. Their C calls the integer-only FP of cpu_helpers.c
   (snx_fp_*), in the runtime library, and writes the result with SETSB/SETDB/SETH.
   The sections below are included into this single C translation unit: no function-pointer dispatch, cross-module
   ABI or runtime overhead. */
#include "common/cpu/recompiler/llvm/fp_paths.h"
#include <string.h>

typedef LLVMValueRef V;
#define B        ir->b
#define F(lo, n) ((w >> (lo)) & ((1UL << (n)) - 1))

#include "common/cpu/recompiler/llvm/fp/guarded.inc"
#include "common/cpu/recompiler/llvm/fp/precise_bridge.inc"
#include "common/cpu/recompiler/llvm/fp/scalar_arithmetic.inc"
#include "common/cpu/recompiler/llvm/fp/scalar_control.inc"
