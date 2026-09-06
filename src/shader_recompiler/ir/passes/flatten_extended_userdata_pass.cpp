// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#include "common/arch.h"
#include "common/decoder.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/emulator_settings.h"
#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"

#ifdef ARCH_X86_64

using namespace Xbyak::util;

static Xbyak::CodeGenerator g_srt_codegen(32_MB);
static const u8* g_srt_codegen_start = nullptr;

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    return func_addr;
}

} // namespace Shader

namespace {

static void DumpSrtProgram(const Shader::Info& info, const u8* code, size_t codesize) {
    using namespace Common::FS;

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}_{:#018x}.srtprogram.txt", info.stage, info.pgm_hash);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create, FileType::TextFile};

    u64 address = reinterpret_cast<u64>(code);
    u64 code_end = address + codesize;
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = ZYAN_STATUS_SUCCESS;
    while (address < code_end && ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                                     instruction, operands, reinterpret_cast<void*>(address)))) {
        std::string s =
            Common::Decoder::Instance()->disassembleInst(instruction, operands, address);
        s += "\n";
        file.WriteString(s);
        address += instruction.length;
    }
}

static bool SrtWalkerSignalHandler(void* context, void* fault_address) {
    // Only handle if the fault address is within the SRT code range
    const u8* code_start = g_srt_codegen_start;
    const u8* code_end = code_start + g_srt_codegen.getSize();
    const void* code = Common::GetRip(context);
    if (code < code_start || code >= code_end) {
        return false; // Not in SRT code range
    }

    // Patch instruction to zero register
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                       const_cast<void*>(code), 15);

    ASSERT(ZYAN_SUCCESS(status) && instruction.mnemonic == ZYDIS_MNEMONIC_MOV &&
           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
           operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY);

    size_t len = instruction.length;
    const size_t patch_size = 3;
    u8* code_patch = const_cast<u8*>(reinterpret_cast<const u8*>(code));

    // We can only encounter rdi or r10d as the first operand in a
    // fault memory access for SRT walker.
    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        // mov rdi, [rdi + (off_dw << 2)] -> xor rdi, rdi
        code_patch[0] = 0x48;
        code_patch[1] = 0x31;
        code_patch[2] = 0xFF;
        break;
    case ZYDIS_REGISTER_R10D:
        // mov r10d, [rdi + (off_dw << 2)] -> xor r10d, r10d
        code_patch[0] = 0x45;
        code_patch[1] = 0x31;
        code_patch[2] = 0xD2;
        break;
    default:
        UNREACHABLE_MSG("Unsupported register for SRT walker patch");
        return false;
    }

    // Fill nops
    memset(code_patch + patch_size, 0x90, len - patch_size);

    LOG_WARNING(Render_Recompiler, "Patched SRT walker at {}, fault address {}", code,
                fault_address);

    return true;
}

using namespace Shader;

struct PassInfo {
    // map offset to inst
    using PtrUserList = boost::container::flat_map<u32,
        boost::container::small_vector<Shader::IR::Inst*, 2>>;

    Optimization::SrtGvnTable gvn_table;
    // keys are GetUserData or ReadConst instructions that are used as pointers
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    // GetUserData instructions corresponding to sgpr_base of SRT roots
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> srt_roots;

    // pick a single inst for a given value number
    std::unordered_map<u32, IR::Inst*> vn_to_inst;

    // Bumped during codegen to assign offsets to readconsts
    u32 dst_off_dw;

    PtrUserList* GetUsesAsPointer(IR::Inst* inst) {
        auto it = pointer_uses.find(inst);
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Return a single instruction that this instruction is identical to, according
    // to value number
    // The "original" is arbitrary. Here it's the first instruction found for a given value number
    IR::Inst* DeduplicateInstruction(IR::Inst* inst) {
        auto it = vn_to_inst.try_emplace(gvn_table.GetValueNumber(inst), inst);
        return it.first->second;
    }
};
} // namespace

namespace Shader::Optimization {

namespace {

static inline void PushPtr(Xbyak::CodeGenerator& c, u32 off_dw) {
    c.push(rdi);
    c.mov(rdi, ptr[rdi + (off_dw << 2)]);
    c.mov(r10, 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
}

static inline void PopPtr(Xbyak::CodeGenerator& c) {
    c.pop(rdi);
};

static void VisitPointer(u32 off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         Xbyak::CodeGenerator& c) {
    PushPtr(c, off_dw);
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    // First copy all the src data from this tree level
    // That way, all data that was contiguous in the guest SRT is also contiguous in the
    // flattened buffer.
    // TODO src and dst are contiguous. Optimize with wider loads/stores
    // TODO if this subtree is dynamically indexed, don't compact it (keep it sparse)
    for (auto& [src_off_dw, uses] : *use_list) {
        c.mov(r10d, ptr[rdi + (src_off_dw << 2)]);
        c.mov(ptr[rsi + (pass_info.dst_off_dw << 2)], r10d);

        for (auto* use : uses) {
            use->SetFlags<u32>(pass_info.dst_off_dw);
        }
        pass_info.dst_off_dw++;
    }

    // Then visit any children used as pointers
    for (const auto& [src_off_dw, uses] : *use_list) {
        for (auto* use : uses) {
            if (pass_info.GetUsesAsPointer(use)) {
                VisitPointer(src_off_dw, use, pass_info, c);
            }
        }
    }

    PopPtr(c);
}

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.srt_roots.empty()) {
        return;
    }

    // Register the signal handler for SRT walker, if not already registered
    if (g_srt_codegen_start == nullptr) {
        g_srt_codegen_start = c.getCurr();
        auto* signals = Core::Signals::Instance();
        // Call after the memory invalidation handler
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    }

    info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);

    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        VisitPointer(static_cast<u32>(sgpr_base), root, pass_info, c);
    }

    c.ret();
    c.ready();

    info.srt_info.walker_func_size =
        c.getCurr() - reinterpret_cast<const u8*>(info.srt_info.walker_func);

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

// Extract "GetUserData root + IAdd32 immediate offset (bytes)" from the single-level shape that
// s_add_u32 s4, s0, <imm> produces: IAdd32(GetUserData, Imm). Deeper or non-matching shapes return
// {nullptr, 0} and the caller falls back to BFS.
//
// NOTE: matches only this known single-level pattern (experimental). A complete solution would
// track the immediate's constant expression at translation time instead of pattern-matching the
// IR shape afterwards.
std::pair<IR::Inst*, u32> TraceUserDataBase(IR::Inst* inst) {
    if (!inst || inst->GetOpcode() != IR::Opcode::IAdd32) {
        return {nullptr, 0};
    }
    const bool imm0 = inst->Arg(0).IsImmediate();
    const bool imm1 = inst->Arg(1).IsImmediate();
    if (imm0 == imm1) {
        return {nullptr, 0}; // both immediate or both non-immediate -> cannot resolve statically
    }
    IR::Inst* base = (imm0 ? inst->Arg(1) : inst->Arg(0)).InstRecursive();
    if (!base || base->GetOpcode() != IR::Opcode::GetUserData) {
        return {nullptr, 0};
    }
    return {base, imm0 ? inst->Arg(0).U32() : inst->Arg(1).U32()};
}

// Check whether the high-32 IAdd32 chain carries a non-zero immediate offset (s_addc_u32 s5, s1,
// <imm>, imm != 0). The high 32 bits are typically IAdd32(IAdd32(GetUserData, imm), carry), i.e.
// exactly one level of IAdd32 nesting; only immediate operands are inspected, so the carry's
// Select(GetScc(),...) is never misread as an offset.
bool HasNonZeroImmediateOffset(IR::Inst* inst) {
    if (!inst || inst->GetOpcode() != IR::Opcode::IAdd32) {
        return false;
    }
    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->Arg(i);
        if (arg.IsImmediate()) {
            if (arg.U32() != 0) {
                return true;
            }
        } else if (IR::Inst* sub = arg.InstRecursive();
                   sub && sub->GetOpcode() == IR::Opcode::IAdd32) {
            for (size_t j = 0; j < sub->NumArgs(); j++) {
                const IR::Value a = sub->Arg(j);
                if (a.IsImmediate() && a.U32() != 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

}; // namespace

void FlattenExtendedUserdataPass(IR::Program& program) {
    Shader::Info& info = program.info;
    PassInfo pass_info;

    // traverse at end and assign offsets to duplicate readconsts, using
    // vn_to_inst as the source
    boost::container::small_vector<IR::Inst*, 32> all_readconsts;

    for (auto r_it = program.post_order_blocks.rbegin(); r_it != program.post_order_blocks.rend();
         r_it++) {
        IR::Block* block = *r_it;
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::ReadConst) {
                if (!inst.Arg(1).IsImmediate()) {
                    LOG_WARNING(Render_Recompiler, "ReadConst has non-immediate offset");
                    continue;
                }

                all_readconsts.push_back(&inst);
                if (pass_info.DeduplicateInstruction(&inst) != &inst) {
                    // This is a duplicate of a readconst we've already visited
                    continue;
                }

                IR::Inst* ptr_composite = inst.Arg(0).InstRecursive();

                const auto pred = [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                    if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                        inst->GetOpcode() == IR::Opcode::ReadConst) {
                        return inst;
                    }
                    return std::nullopt;
                };

                // Recognize "address = consecutive-SGPR GetUserData root + immediate offset":
                //   low 32 -> GetUserData(N), high 32 -> GetUserData(N+1), N/N+1 consecutive.
                // Only accumulate the offset for this shape (the s_add_u32 s4, s0, <imm> immediate);
                // otherwise fall back to the original BFS with offset 0 (avoid corrupting
                // multi-level pointers / dynamic indexing / non-consecutive SGPRs).
                auto [lo_base, lo_byte_offset] =
                    TraceUserDataBase(ptr_composite->Arg(0).InstRecursive());
                auto hi_base = IR::BreadthFirstSearch(ptr_composite->Arg(1), pred);

                IR::Inst* ptr_lo = nullptr;
                u32 byte_offset = 0;
                if (lo_base && hi_base && lo_base->GetOpcode() == IR::Opcode::GetUserData &&
                    hi_base.value()->GetOpcode() == IR::Opcode::GetUserData &&
                    static_cast<u32>(lo_base->Arg(0).ScalarReg()) + 1 ==
                        static_cast<u32>(hi_base.value()->Arg(0).ScalarReg())) {
                    // Consecutive SGPR root: the low-32 offset is the address byte offset.
                    // A non-zero high-32 immediate offset (s_addc_u32 s5, s1, <imm>, imm != 0) is
                    // statically detectable, so warn instead of silently dropping it. Carry overflow
                    // (Select(GetScc(),1,0)) cannot be accumulated statically; treat it as 0 (the
                    // carry window is only 272 bytes, effectively always 0).
                    if (HasNonZeroImmediateOffset(ptr_composite->Arg(1).InstRecursive())) {
                        LOG_ERROR(Render_Recompiler,
                                  "SRT flatten: high 32-bit pointer has non-zero immediate "
                                  "offset, dropped by 32-bit offset tracing");
                    }
                    ptr_lo = lo_base;
                    byte_offset = lo_byte_offset;
                } else {
                    // Fall back to the original BFS behavior
                    auto base0 = IR::BreadthFirstSearch(ptr_composite->Arg(0), pred);
                    auto base1 = IR::BreadthFirstSearch(ptr_composite->Arg(1), pred);
                    ASSERT_MSG(base0 && base1, "ReadConst not from constant memory");
                    ptr_lo = base0.value();
                    byte_offset = 0;
                }

                ptr_lo = pass_info.DeduplicateInstruction(ptr_lo);

                auto ptr_uses_kv =
                    pass_info.pointer_uses.try_emplace(ptr_lo, PassInfo::PtrUserList{});
                PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;

                // Byte offset -> dwords, added to the ReadConst offset (SMRD offset field is in dwords)
                if (byte_offset & 3) {
                    LOG_ERROR(Render_Recompiler,
                              "SRT flatten: pointer byte offset {:#x} not dword-aligned, low 2 "
                              "bits dropped",
                              byte_offset);
                }
                const u32 src_off_dw = inst.Arg(1).U32() + (byte_offset >> 2);
                user_list[src_off_dw].push_back(&inst);

                if (ptr_lo->GetOpcode() == IR::Opcode::GetUserData) {
                    IR::ScalarReg ud_reg = ptr_lo->Arg(0).ScalarReg();
                    pass_info.srt_roots[ud_reg] = ptr_lo;
                }
            }
        }
    }

    GenerateSrtProgram(info, pass_info);

    // Assign offsets to duplicate readconsts
    for (IR::Inst* readconst : all_readconsts) {
        ASSERT(pass_info.vn_to_inst.contains(pass_info.gvn_table.GetValueNumber(readconst)));
        IR::Inst* original = pass_info.DeduplicateInstruction(readconst);
        readconst->SetFlags<u32>(original->Flags<u32>());
    }

    info.RefreshFlatBuf();
}

} // namespace Shader::Optimization

#else

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    UNREACHABLE_MSG("RegisterWalkerCode unimplemented for target architecture.");
}

namespace Optimization {

void FlattenExtendedUserdataPass(IR::Program& program) {
    UNREACHABLE_MSG("FlattenExtendedUserdataPass unimplemented for target architecture.");
}

} // namespace Optimization

} // namespace Shader

#endif
