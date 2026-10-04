// Exercise the supported internal Program transaction contract, independently of Engine's
// boundary-only public cancellation policy. All operands come from the real v3 Model loader.
namespace {
void program_zero_commit(const char* artifact, bool selected_compute,
                         ninfer::runtime::ResidentModelSession* session = nullptr) {
    namespace model = ninfer::models::qwen3_5;
    namespace rt = ninfer::runtime;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 256;
    options.prefill_chunk = 256;
    options.max_concurrency = 2;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens = 15;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ninfer::SpeculativeRoutingProfileIdentity identity;
    std::vector<ninfer::TokenId> prompt, follow;
    ninfer::GenerationResult reference, follow_reference;
    {
        auto fixed = make_engine(options, session);
        identity = *fixed.load_summary().speculative_routing_identity;
        prompt = fixed.tokenize_text("Count from one to twenty: one, two, three,");
    }
    {
        auto baseline_options = options;
        baseline_options.speculative = {};
        auto baseline = make_engine(baseline_options, session);
        reference = baseline.generate(baseline.prepare_tokens(prompt), request());
        require(reference.generated_token_ids.size() == 16, "zero-commit reference ended early");
        follow = prompt;
        follow.push_back(reference.generated_token_ids.front());
        follow_reference = baseline.generate(baseline.prepare_tokens(follow), request());
    }
    for (const auto action : {0U, 7U}) {
        ProfileFile profile;
        nlohmann::json cells = nlohmann::json::array();
        for (unsigned batch = 1; batch <= 2; ++batch) {
            for (const auto upper : {1024, 8192, 32768}) {
                cells.push_back({{"active_batch", batch}, {"frontier_upper", upper},
                                 {"draft_tokens", action}});
            }
        }
        {
            std::ofstream out(profile.path);
            out << routing_profile(identity, cells, selected_compute);
        }
        options.speculative.routing.mode = ninfer::SpeculativeRoutingMode::Calibrated;
        options.speculative.routing.profile_path = profile.path;
        auto normalized = rt::normalize_engine_options(options);
        ninfer::DeviceContext device(0);
        auto constructed = session ? session->make_instance(normalized, device) : rt::construct_model(normalized, device);
        auto& instance = *constructed.instance;
        auto& program = *instance.program;
        rt::ResolvedExecutionOptions execution;
        execution.requested_output_tokens = 32;
        execution.allow_prefix_reuse = false;
        execution.sampling.temperature = 0.0F;
        execution.sampling.presence_penalty = 0.0F;
        execution.sampling.frequency_penalty = 0.0F;
        const auto admit = [&](const std::vector<ninfer::TokenId>& tokens, unsigned lane) {
            auto prepared = instance.frontend.prepare_tokens(tokens);
            const auto base = program.plan_request(prepared, execution);
            auto candidate = program.inspect_admission(prepared, base, rt::LaneId{lane},
                                                       nullptr, nullptr, std::nullopt, false);
            require(candidate.has_value(), "zero-commit cold admission was infeasible");
            auto plan = program.seal_identity(*candidate, prepared, {});
            require(plan.has_value(), "zero-commit cold admission did not seal");
            require(program.start_resource_transaction(std::move(*plan), std::move(prepared), {}) ==
                        rt::ContextTransactionReserveStatus::Reserved,
                    "zero-commit cold admission did not reserve");
            std::optional<model::SequenceHandle> sequence;
            while (!sequence) {
                auto progress = program.progress_context_transaction({});
                if (auto* result = std::get_if<model::MaterializationResult>(&progress)) {
                    require(result->status == rt::ContextTransactionStatus::Published &&
                                result->published.has_value(), "zero-commit admission aborted");
                    sequence = result->published->sequence;
                    program.finalize_context_transaction();
                }
            }
            std::vector<ninfer::TokenId> generated;
            while (generated.empty()) {
                auto progress = program.advance_prefill(*sequence);
                if (progress.capture) { program.skip_capture(std::move(*progress.capture)); }
                if (progress.pending) {
                    generated.push_back(progress.pending->tokens().front());
                    const rt::CommitDecision decision{.accepted_tokens = 1};
                    auto result = program.commit(std::move(*progress.pending), {&decision, 1});
                    for (auto& capture : result.captures) {
                        if (capture) { program.skip_capture(std::move(*capture)); }
                    }
                }
            }
            return std::pair{*sequence, generated};
        };
        // Lane 1 remains owned throughout lane 0's cancellation. Its continuation proves that
        // zero-prefix settlement cannot modify another lane's committed KV/recurrent state.
        auto survivor = admit(prompt, 1);
        const auto survivor_usage = program.physical_usage();
        auto cancelled = admit(prompt, 0);
        require(cancelled.second == survivor.second &&
                    cancelled.second.front() == reference.generated_token_ids.front(),
                "zero-commit begin token differs from target-only");
        const rt::RoundBudget budget{.generated_tokens_remaining = 31};
        auto pending = program.decode({&cancelled.first, 1}, {&budget, 1});
        const auto facts = pending.decode_execution();
        require(facts.draft_tokens == action && facts.verify_width == action + 1 &&
                    facts.proposal_width == proposal_width(action, selected_compute) &&
                    facts.neural_drafter_executed == (action != 0),
                "zero-commit transaction did not execute its real selected action");
        const rt::CommitDecision zero{.accepted_tokens = 0, .terminal = true, .cancelled = true};
        const auto committed = program.commit(std::move(pending), {&zero, 1});
        require(committed.row_count == 1 &&
                    committed.rows[0].disposition == rt::CommitDisposition::CancelledReleased,
                "zero-commit transaction did not release only its cancelled lane");
        const auto after = program.physical_usage();
        require(after.device_state_slots == survivor_usage.device_state_slots &&
                    after.host_state_slots == survivor_usage.host_state_slots &&
                    after.device_main_kv_pages == survivor_usage.device_main_kv_pages &&
                    after.device_backend_kv_pages == survivor_usage.device_backend_kv_pages &&
                    after.host_kv_bytes == survivor_usage.host_kv_bytes,
                "zero-commit cancellation retained or stole physical ownership");
        const auto complete = [&](auto active, const std::vector<ninfer::TokenId>& expected) {
            while (active.second.size() < expected.size()) {
                const rt::RoundBudget remaining{.generated_tokens_remaining =
                    static_cast<unsigned>(expected.size() - active.second.size())};
                auto next = program.decode({&active.first, 1}, {&remaining, 1});
                const auto licensed = next.row_counts().empty() ? 1 : next.row_counts().front();
                require(licensed > 0 && static_cast<unsigned>(licensed) <= next.row_stride() &&
                            static_cast<std::size_t>(licensed) <= next.tokens().size(),
                        "zero-commit continuation returned an invalid licensed token extent");
                const auto count = std::min<std::size_t>(licensed, remaining.generated_tokens_remaining);
                active.second.insert(active.second.end(), next.tokens().begin(), next.tokens().begin() + count);
                const rt::CommitDecision decision{.accepted_tokens = static_cast<unsigned>(count),
                    .terminal = active.second.size() == expected.size()};
                auto result = program.commit(std::move(next), {&decision, 1});
                for (auto& capture : result.captures) {
                    if (capture) { program.skip_capture(std::move(*capture)); }
                }
            }
            require(active.second == expected, "zero-commit settlement changed subsequent target tokens");
            auto finished = program.finish(active.first);
            require(finished.status == rt::ConsumeStatus::Consumed, "zero-commit continuation did not finish");
            if (finished.continuation) {
                (void)program.release_continuation(std::move(*finished.continuation));
            }
        };
        complete(survivor, reference.generated_token_ids);
        complete(admit(follow, 0), follow_reference.generated_token_ids);
        std::cout << "ok Program executed zero-prefix commit K=" << action
                  << ": cancelled ownership released, owned survivor and re-admission exact\n";
    }
}
} // namespace
