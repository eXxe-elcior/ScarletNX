/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* The native Maxwell translator must reuse unconditional SSA values, but never
   forward one through an instruction guarded by a Maxwell predicate. */
#include "nx/gpu/shader/shader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SCHEDULE_WORD 0x001F8000FC0007E0ull
#define EXIT_WORD     0xE30000000007000Full
#define FLOAT_MUL     0x5C68000000070000ull
#define ATTRIBUTE_OUT 0xEFF0000000070000ull

static u64 immediate_move(u32 register_index, u32 value) {
    return 0x0100000000000000ull | ((u64)value << 20) | 0x71000ull | register_index;
}

static unsigned count_spirv_instructions(const u32* words, size_t count, u32 operation) {
    unsigned total = 0;
    size_t index = 5;
    while (index < count) {
        u32 instruction_length = words[index] >> 16;
        if (!instruction_length || index + instruction_length > count) return (unsigned)-1;
        if ((words[index] & 0xffff) == operation) total++;
        index += instruction_length;
    }
    return index == count ? total : (unsigned)-1;
}

static unsigned translate_and_count_register_loads(int conditional, unsigned* selections) {
    u64 code[32] = {0};
    ShaderInfo info = {0};
    size_t word_count = 0;
    u32* spirv;
    unsigned loads;

    code[10] = SCHEDULE_WORD;
    code[11] = immediate_move(1, 0x3f800000u);
    code[12] = immediate_move(2, 0x40000000u);
    /* R3 = R1 * R2; the optional predicate P0 is initially false. */
    code[13] = FLOAT_MUL | (1ull << 8) | (2ull << 20) | 3;
    if (conditional) code[13] &= ~(7ull << 16);
    code[14] = SCHEDULE_WORD;
    code[15] = FLOAT_MUL | (3ull << 8) | (2ull << 20) | 4;
    code[16] = ATTRIBUTE_OUT | (0x70ull << 20) | (255ull << 8) | (4ull << 0);
    code[17] = EXIT_WORD;
    spirv = shader_to_spirv((const u8*)code, sizeof code, ST_VS, &info, &word_count, 1, 1, 1, 1);
    if (!spirv || info.unsupported) {
        fprintf(stderr, "shader register forwarding: %s\n", info.first_unsupported);
        free(spirv);
        return (unsigned)-1;
    }
    loads = count_spirv_instructions(spirv, word_count, 61); /* OpLoad */
    *selections = count_spirv_instructions(spirv, word_count, 247); /* OpSelectionMerge */
    free(spirv);
    return loads;
}

int main(void) {
    unsigned unguarded_selections, guarded_selections;
    unsigned unguarded = translate_and_count_register_loads(0, &unguarded_selections);
    unsigned guarded = translate_and_count_register_loads(1, &guarded_selections);
    if (unguarded == (unsigned)-1 || guarded == (unsigned)-1 || guarded_selections <= unguarded_selections ||
        guarded <= unguarded) {
        fprintf(stderr, "FAIL: register forwarding: unguarded loads=%u, guarded loads=%u, selections=%u/%u\n",
                unguarded, guarded, unguarded_selections, guarded_selections);
        return 1;
    }
    printf("shader register forwarding: unguarded=%u OpLoad, guarded=%u OpLoad (safe merge)\n", unguarded,
           guarded);
    return 0;
}
