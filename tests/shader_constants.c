/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright 2026 the ScarletNX contributors */
/* Constant interning must be bit-exact, type-exact, and scalable. */
#include "nx/gpu/shader/spirv_emit.h"
#include <stdint.h>
#include <stdio.h>

#define REQUIRE(test) do { if (!(test)) { \
    fprintf(stderr, "shader constants: failed at line %d: %s\n", __LINE__, #test); \
    spv_emit_destroy(&emitter); return 1; \
} } while (0)

int main(void) {
    SpvEmitter emitter;
    spv_emit_init(&emitter, 0);
    uint32_t unsigned_type = spv_emit_type_u32(&emitter);
    uint32_t signed_type = spv_emit_type_i32(&emitter);
    uint32_t float_type = spv_emit_type_f32(&emitter);
    uint32_t bool_type = spv_emit_type_bool(&emitter);
    uint32_t double_type = spv_emit_type_f64(&emitter);
    uint32_t vec4_type = spv_emit_type_vec4(&emitter, float_type);
    uint32_t vec3_type = spv_emit_type_vec3(&emitter, float_type);
    uint32_t vec2_type = spv_emit_type_vec2(&emitter, float_type);

    uint32_t integer = spv_emit_constant_u32(&emitter, unsigned_type, 0x80000000u);
    REQUIRE(integer == spv_emit_constant_u32(&emitter, unsigned_type, 0x80000000u));
    REQUIRE(integer != spv_emit_constant_u32(&emitter, signed_type, 0x80000000u));
    REQUIRE(integer != spv_emit_constant_u32(&emitter, float_type, 0x80000000u));
    REQUIRE(spv_emit_constant_f32(&emitter, float_type, 0.0f) !=
            spv_emit_constant_f32(&emitter, float_type, -0.0f));
    REQUIRE(spv_emit_constant_bool(&emitter, bool_type, 1) ==
            spv_emit_constant_bool(&emitter, bool_type, 4));
    REQUIRE(spv_emit_constant_bool(&emitter, bool_type, 0) !=
            spv_emit_constant_bool(&emitter, bool_type, 1));
    uint32_t large_value = spv_emit_constant64(&emitter, double_type, UINT64_C(0x3ff0000000000000));
    REQUIRE(large_value == spv_emit_constant64(&emitter, double_type, UINT64_C(0x3ff0000000000000)));
    REQUIRE(large_value != spv_emit_constant64(&emitter, double_type, UINT64_C(0x000000003ff00000)));

    uint32_t c1 = spv_emit_constant_f32(&emitter, float_type, 1.0f);
    uint32_t c2 = spv_emit_constant_f32(&emitter, float_type, 2.0f);
    uint32_t c3 = spv_emit_constant_f32(&emitter, float_type, 3.0f);
    uint32_t c4 = spv_emit_constant_f32(&emitter, float_type, 4.0f);
    uint32_t components[4] = {c1, c2, c3, c4};
    uint32_t vector = spv_emit_constant_vec4(&emitter, vec4_type, components);
    REQUIRE(vector == spv_emit_constant_vec4(&emitter, vec4_type, components));
    REQUIRE(spv_emit_constant_vec2(&emitter, vec2_type, c1, c2) ==
            spv_emit_constant_vec2(&emitter, vec2_type, c1, c2));
    REQUIRE(spv_emit_constant_vec3(&emitter, vec3_type, c1, c2, c3) ==
            spv_emit_constant_vec3(&emitter, vec3_type, c1, c2, c3));
    components[3] = c3;
    REQUIRE(vector != spv_emit_constant_vec4(&emitter, vec4_type, components));

    /* Rehash many times, including keys colliding in their low hash bits. */
    for (uint32_t index = 0; index < 4096; index++)
        REQUIRE(spv_emit_constant_u32(&emitter, unsigned_type, index + 100) != 0);
    size_t constant_count = emitter.constant_count;
    size_t types_words = emitter.types.count;
    for (uint32_t iteration = 0; iteration < 16; iteration++)
        for (uint32_t index = 0; index < 4096; index++)
            REQUIRE(spv_emit_constant_u32(&emitter, unsigned_type, index + 100) != 0);
    REQUIRE(emitter.constant_count == constant_count);
    REQUIRE(emitter.types.count == types_words);
    printf("shader constants: %zu unique, 65,536 repeated lookups, zero duplicate declarations\n",
           emitter.constant_count);
    spv_emit_destroy(&emitter);
    return 0;
}
