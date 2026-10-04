#include "models/load_options.h"

#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;
using models::LoadOptions;

bool compatible(const LoadOptions& loaded, const LoadOptions& requested) {
    return models::execution_load_options_compatible(loaded, requested);
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() try {
    LoadOptions loaded;
    loaded.speculative = SpeculativeBackend::DFlash2;
    require(compatible(loaded, loaded), "same execution configuration rejected");
    auto target = loaded;
    target.speculative = SpeculativeBackend::None;
    require(compatible(loaded, target), "resident DFlash2 Full cannot execute target-only");
    require(!compatible(target, loaded), "target-only weights admitted a missing drafter");
    auto changed = target;
    changed.prefill_a8 = !loaded.prefill_a8;
    require(!compatible(loaded, changed), "different target arithmetic admitted");
    changed = target;
    changed.lm_head_q4 = !loaded.lm_head_q4;
    require(!compatible(loaded, changed), "different stored target representation admitted");
    changed = target;
    changed.proposal_head = ProposalHead::Optimized;
    require(!compatible(loaded, changed), "different proposal representation admitted");
    auto unsupported = loaded;
    unsupported.speculative = SpeculativeBackend::Mtp;
    require(!compatible(unsupported, target), "MTP mismatch admitted as resident DFlash2");
    unsupported = loaded;
    unsupported.vision = true;
    changed = unsupported;
    changed.speculative = SpeculativeBackend::None;
    require(!compatible(unsupported, changed), "Vision backing admitted target-only resident borrowing");
    unsupported = loaded;
    unsupported.ranks = 2;
    changed = unsupported;
    changed.speculative = SpeculativeBackend::None;
    require(!compatible(unsupported, changed), "multi-GPU backing admitted resident borrowing");
    auto a16 = target;
    a16.prefill_a8 = false;
    require(models::resident_load_options_compatible(loaded, a16),
            "native A16 preparation rejected same stored operands");
    require(models::prepared_execution_options_compatible(loaded, false, a16),
            "actual A16 preparation rejected requested A16 execution");
    require(!models::prepared_execution_options_compatible(loaded, true, a16),
            "A8 Parameters admitted requested A16 arithmetic");
    a16.prefill_cublas = !loaded.prefill_cublas;
    require(!models::resident_load_options_compatible(loaded, a16),
            "resident admission ignored unrelated arithmetic drift");
    require(!models::prepared_execution_options_compatible(loaded, false, a16),
            "prepared admission ignored unrelated arithmetic drift");
    std::cout << "resident execution compatibility passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
