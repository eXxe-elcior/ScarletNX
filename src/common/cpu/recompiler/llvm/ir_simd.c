/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors
   Ported in part from QEMU (target/arm/tcg/translate-a64.c, helper-a64.c, vec_helper.c, vfp_helper.c,
   crypto_helper.c), LGPL-2.1-or-later:
   Copyright (c) 2013 Alexander Graf <agraf@suse.de>
   Copyright (c) 2018 Linaro
   Copyright (C) 2013 - 2018 Linaro Ltd <ard.biesheuvel@linaro.org>
   Copyright (c) 2003 Fabrice Bellard */
/* AdvSIMD: snx_a64_fast_simd's shortcuts (a64.c), emitted inline as the C it prints, and the SIMD patterns, whose
   template hands the word to snx_simd (cpu_helpers.c, in the runtime library).
   The sections below are included into this single C translation unit: no function-pointer dispatch, cross-module
   ABI or runtime overhead. */
#include "common/cpu/recompiler/llvm/fp_paths.h"
#include <string.h>

typedef LLVMValueRef V;
#define B ir->b

#include "common/cpu/recompiler/llvm/neon/core.inc"
#include "common/cpu/recompiler/llvm/neon/layout_memory.inc"
#include "common/cpu/recompiler/llvm/neon/crypto_bytes.inc"
#include "common/cpu/recompiler/llvm/neon/integer.inc"
#include "common/cpu/recompiler/llvm/neon/floating.inc"
#include "common/cpu/recompiler/llvm/neon/misc.inc"
#include "common/cpu/recompiler/llvm/neon/decode.inc"
