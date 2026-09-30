#include "models/qwen3_5/program/speculative/lookup_vote.h"
#include <iostream>
#include <vector>

static int check(bool ok, const char* msg) { if(ok) return 0; std::cerr<<msg<<'\n'; return 1; }

int main() {
    using namespace ninfer;
    using namespace ninfer::qwen3_5;
    int fail=0;
    LookupDraftOptions o;
    o.max_order=4; o.max_matches=16; o.min_support=1; o.min_confidence=0.5F;
    std::vector<TokenId> ledger{1,2,3,4,8, 1,2,3,4,8, 1,2,3,4};
    auto p=lookup_draft_vote(ledger,4,7,o,nullptr,nullptr);
    fail+=check(p && p.match_order==4 && p.tokens[0]==8 && p.support==2,
                "local voting did not combine repeated continuation");

    LookupPersistentStore store(256,4,16);
    std::vector<TokenId> prior{4,5,6,7,42,43};
    store.append_sequence(prior);
    std::vector<TokenId> query{10,4,5,6,7};
    o.min_support=1; o.min_confidence=0.5F;
    auto q=lookup_draft_vote(query,4,2,o,&store,nullptr);
    fail+=check(q && q.tokens[0]==42 && q.tokens[1]==43 && (q.sources&2)!=0,
                "persistent suffix history did not propose continuation");

    std::array<TokenId,4> recent{};
    const auto n=lookup_draft(std::span<const TokenId>(ledger),4,4,recent.data());
    fail+=check(n!=0 && recent[0]==8,"historical recent lookup changed semantics");
    return fail?1:0;
}
