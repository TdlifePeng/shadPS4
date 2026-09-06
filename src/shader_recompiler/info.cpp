// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "common/debug.h"
#include "common/hack_features.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace Shader {

namespace {

// Build a bitmap of valid data formats once (data_format is 6 bits, values 0~63, fits in a u64).
constexpr u64 MakeValidDataFormatMask() {
    // The valid data format values: every value defined in the DataFormat enum except FormatInvalid
    // and the reserved gaps (15, 23~31, 42~46 in GCN Table 8.13). FMask values (47~56) are real,
    // defined formats and are kept here — they are excluded separately by AmdGpu::IsFmask.
    constexpr std::array kValidDataFormats = {
        AmdGpu::DataFormat::Format8,          AmdGpu::DataFormat::Format16,
        AmdGpu::DataFormat::Format8_8,        AmdGpu::DataFormat::Format32,
        AmdGpu::DataFormat::Format16_16,      AmdGpu::DataFormat::Format10_11_11,
        AmdGpu::DataFormat::Format11_11_10,   AmdGpu::DataFormat::Format10_10_10_2,
        AmdGpu::DataFormat::Format2_10_10_10, AmdGpu::DataFormat::Format8_8_8_8,
        AmdGpu::DataFormat::Format32_32,      AmdGpu::DataFormat::Format16_16_16_16,
        AmdGpu::DataFormat::Format32_32_32,   AmdGpu::DataFormat::Format32_32_32_32,
        AmdGpu::DataFormat::Format5_6_5,      AmdGpu::DataFormat::Format1_5_5_5,
        AmdGpu::DataFormat::Format5_5_5_1,    AmdGpu::DataFormat::Format4_4_4_4,
        AmdGpu::DataFormat::Format8_24,       AmdGpu::DataFormat::Format24_8,
        AmdGpu::DataFormat::FormatX24_8_32,   AmdGpu::DataFormat::FormatGB_GR,
        AmdGpu::DataFormat::FormatBG_RG,      AmdGpu::DataFormat::Format5_9_9_9,
        AmdGpu::DataFormat::FormatBc1,        AmdGpu::DataFormat::FormatBc2,
        AmdGpu::DataFormat::FormatBc3,        AmdGpu::DataFormat::FormatBc4,
        AmdGpu::DataFormat::FormatBc5,        AmdGpu::DataFormat::FormatBc6,
        AmdGpu::DataFormat::FormatBc7,        AmdGpu::DataFormat::FormatFmask8_1,
        AmdGpu::DataFormat::FormatFmask8_2,   AmdGpu::DataFormat::FormatFmask8_4,
        AmdGpu::DataFormat::FormatFmask16_1,  AmdGpu::DataFormat::FormatFmask16_2,
        AmdGpu::DataFormat::FormatFmask32_2,  AmdGpu::DataFormat::FormatFmask32_4,
        AmdGpu::DataFormat::FormatFmask32_8,  AmdGpu::DataFormat::FormatFmask64_4,
        AmdGpu::DataFormat::FormatFmask64_8,  AmdGpu::DataFormat::Format4_4,
        AmdGpu::DataFormat::Format6_5_5,      AmdGpu::DataFormat::Format1,
        AmdGpu::DataFormat::Format1_Reversed, AmdGpu::DataFormat::Format32_As_8,
        AmdGpu::DataFormat::Format32_As_8_8,  AmdGpu::DataFormat::Format32_As_32_32_32_32,
    };

    u64 mask = 0;
    for (const AmdGpu::DataFormat fmt : kValidDataFormats) {
        mask |= u64{1} << static_cast<u32>(fmt);
    }
    return mask;
}

constexpr u64 kValidDataFormatMask = MakeValidDataFormatMask();

// Whether data_format is a real format defined in the enum (not FormatInvalid, not a reserved gap).
bool IsValidDataFormat(const AmdGpu::DataFormat fmt) {
    const u32 value = static_cast<u32>(fmt);
    return value < (sizeof(kValidDataFormatMask) * 8) && ((kValidDataFormatMask >> value) & 1u) != 0;
}

} // namespace

// Clears is_srt_offset T#s in the freshly-filled flatbuf whose base address is not a valid guest
// mapping (or whose array range is invalid), so garbage probe slots never reach texture cache nor
// vary the pipeline specialization.
void Info::PostRefreshFlatBuf() {
    ZoneScopedN("PostRefreshFlatBuf");
    auto* memory = Core::Memory::Instance();
    for (const auto& image : images) {
        if (!image.is_srt_offset) {
            continue;
        }
        auto& sharp = *reinterpret_cast<AmdGpu::Image*>(&flattened_ud_buf[image.sharp_idx]);
        const auto data_format = static_cast<AmdGpu::DataFormat>(sharp.data_format);
        {
            if (AmdGpu::IsFmask(data_format) || !IsValidDataFormat(data_format)) {
                goto clean;
            }
        }
        {
            const auto image_type = static_cast<AmdGpu::ImageType>(sharp.type);
            const bool is_array = image_type == AmdGpu::ImageType::Color1DArray ||
                                  image_type == AmdGpu::ImageType::Color2DArray ||
                                  image_type == AmdGpu::ImageType::Color2DMsaaArray ||
                                  image_type == AmdGpu::ImageType::Cube;
            if (is_array) {
                if (sharp.last_array < sharp.base_array || sharp.depth < sharp.last_array) {
                    goto clean;
                }
            } else {
                if (sharp.base_array != 0 || sharp.last_array != 0) {
                    goto clean;
                }
            }
        }
        // clear any T# whose base address is not a valid guest mapping.
        if (!memory->IsAddressMapped(sharp.Address()))
            goto clean;

        // clear the rest until per-shader patterns are found in the logs.
        if (Common::HackFeatures::isTheOrder1886) {
            // too many shader variants
            if (((AmdGpu::ImageType)sharp.type) != AmdGpu::ImageType::Color3D)
                goto clean;
            if (pgm_hash == 0x804aeead && data_format != AmdGpu::DataFormat::FormatBc4 &&
                data_format != AmdGpu::DataFormat::Format10_11_11)
                goto clean;
            //if (pgm_hash == 0xc75dd317 || pgm_hash == 0x8c8a4817 || pgm_hash == 0x804aeead)
                continue;
        }

    clean:
        sharp = AmdGpu::Image::Null(image.is_depth);
    }
}

} // namespace Shader
