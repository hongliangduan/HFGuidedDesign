import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from collections import defaultdict
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from highfold import (
    cyclepeptide_protein,
    linearpeptide_protein,
    sspeptide_protein,
    extract_sequence_from_pdb,
)
from models.permeability.permeable import permeable
from Bio.Align import substitution_matrices



class ScoreCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache



class IndependentMutationStrategy:
    def __init__(self, aa_vocab, c=2.0):
        self.aa_vocab = aa_vocab
        self.c = c  # exploration constant


        self.individual_mutation_stats = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "attempts": 0,
                    "total_reward": 0.0,
                    "successes": 0,
                }
            )
        )


        self.individual_total_attempts = defaultdict(int)

    def get_beneficial_mutations(self, original_aa, sequence_id, n_select=2):

        candidates = []


        seq_stats = self.individual_mutation_stats[sequence_id]
        seq_total_attempts = self.individual_total_attempts[sequence_id]

        for alt_aa in self.aa_vocab:
            if alt_aa == original_aa:
                continue

            key = (original_aa, alt_aa)
            stats = seq_stats[key]

            if stats["attempts"] == 0:

                score = float("inf")
            else:

                avg_reward = stats["total_reward"] / stats["attempts"]
                exploration = self.c * np.sqrt(
                    np.log(seq_total_attempts + 1) / stats["attempts"]
                )
                score = avg_reward + exploration

            candidates.append((alt_aa, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [aa for aa, _ in candidates[:n_select]]

    def update_mutation_history(self, original_aa, mutated_aa, reward, sequence_id):

        key = (original_aa, mutated_aa)


        stats = self.individual_mutation_stats[sequence_id][key]
        stats["attempts"] += 1
        stats["total_reward"] += reward


        if reward > 0:
            stats["successes"] += 1


        self.individual_total_attempts[sequence_id] += 1

    def get_sequence_stats_summary(self, sequence_id):

        seq_stats = self.individual_mutation_stats[sequence_id]
        total_attempts = self.individual_total_attempts[sequence_id]

        if not seq_stats:
            return f"Sequence {sequence_id}: No mutations attempted yet"

        total_successes = sum(stats["successes"] for stats in seq_stats.values())
        success_rate = total_successes / total_attempts if total_attempts > 0 else 0

        return f"Sequence {sequence_id}: {total_attempts} attempts, {total_successes} successes, {success_rate:.2%} success rate"


class IndividualBestTracker:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.best_sequences = [None] * batch_size
        self.best_scores = [float("-inf")] * batch_size
        self.best_timesteps = [None] * batch_size


        self.sequence_histories = [[] for _ in range(batch_size)]
        self.score_histories = [[] for _ in range(batch_size)]

    def update(self, timestep, sequences, scores):

        for seq_id in range(self.batch_size):
            if seq_id < len(sequences) and seq_id < len(scores):
                current_seq = sequences[seq_id]
                current_score = scores[seq_id]


                self.sequence_histories[seq_id].append(current_seq)
                self.score_histories[seq_id].append(current_score)

                if (
                    current_score is not None
                    and current_score > self.best_scores[seq_id]
                ):
                    self.best_sequences[seq_id] = current_seq
                    self.best_scores[seq_id] = current_score
                    self.best_timesteps[seq_id] = timestep

    def get_results(self):
        results = []
        for seq_id in range(self.batch_size):
            results.append(
                {
                    "sequence_id": seq_id,
                    "best_sequence": self.best_sequences[seq_id],
                    "best_score": self.best_scores[seq_id],
                    "best_timestep": self.best_timesteps[seq_id],
                    "sequence_history": self.sequence_histories[seq_id],
                    "score_history": self.score_histories[seq_id],
                }
            )
        return results


def hybrid_position_selection(
    sample,
    prediction,
    tokenizer,
    fixed_positions,
    n_mutate,
    mutation_strategy,
    sequence_id,  
    timestep,
    timesteps,
    entropy_weight=1.0,
    gap_weight=0.5,
    exploration_noise=0.2,
):
    seq_len = len(sample)
    progress = 1 - timestep / timesteps

    with torch.no_grad():
        if prediction.dim() == 3:
            prediction = prediction[0]

        logits = prediction[:, : tokenizer.K]
        logits_standard = logits[:, :20]
        probs = F.softmax(logits_standard, dim=-1)

        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [L]
        top_probs, _ = torch.topk(probs, k=2, dim=-1)
        confidence_gap = top_probs[:, 0] - top_probs[:, 1]


        mutation_score = entropy_weight * entropy - gap_weight * confidence_gap


        current_seq = tokenizer.untokenize(sample)
        seq_stats = mutation_strategy.individual_mutation_stats[sequence_id]

        for i in range(seq_len):
            if i in fixed_positions:
                continue
            current_aa = current_seq[i]

            total_successes = 0
            total_attempts = 0
            for (orig_aa, mut_aa), stats in seq_stats.items():
                if orig_aa == current_aa:
                    total_successes += stats["successes"]
                    total_attempts += stats["attempts"]

            if total_attempts > 0:
                success_rate = total_successes / total_attempts
                mutation_score[i] *= 1 + success_rate

        if progress >= 0.6:
            mutation_score += exploration_noise * torch.randn_like(mutation_score)

        for idx in fixed_positions:
            mutation_score[idx] = -1e6

        available_positions = [i for i in range(seq_len) if i not in fixed_positions]
        actual_n_mutate = min(n_mutate, len(available_positions))

        if actual_n_mutate > 0:
            _, selected_positions = torch.topk(mutation_score, actual_n_mutate)
            return selected_positions.tolist()
        else:
            return []


def adaptive_guidance_gradient_with_accumulation(
    sample,
    tokenizer,
    guidance_type,
    cache,
    target_protein,
    chain_id,
    model_runner,
    mutation_strategy,
    sequence_id, 
    fixed_positions,
    guidance_scale,
    timestep,
    timesteps,
    base_score_smooth=None,
    score_momentum=0.9,
    prediction=None,
    scheduler=None,
    target_score=3.0,
    peptide_type="cycle",
):
    seq_len = len(sample)

    current_seq = tokenizer.untokenize(sample)
    base_score = batch_evaluate_sequences(
        [current_seq],
        guidance_type,
        cache,
        target_protein,
        chain_id,
        model_runner,
        peptide_type,
    )[0]

    if base_score_smooth is None:
        base_score_smooth = base_score
    else:
        base_score_smooth = (
            score_momentum * base_score_smooth + (1 - score_momentum) * base_score
        )

    if scheduler is not None:
        dynamic_guidance_scale = scheduler.get_guidance_scale(
            timestep, timesteps, base_score_smooth
        )
    else:
        dynamic_guidance_scale = guidance_scale * (timestep / timesteps)

    grads = torch.zeros(seq_len, tokenizer.K - 6, device=sample.device)

    distance_to_target = abs(base_score_smooth - target_score)

    if base_score_smooth >= target_score:
        n_mutate = max(1, seq_len // 10)
        n_alts_per_pos = 1
    elif distance_to_target > 1.0:
        n_mutate = max(4, seq_len // 2)
        n_alts_per_pos = 2
    elif distance_to_target > 0.5:
        n_mutate = max(3, seq_len // 3)
        n_alts_per_pos = 2
    else:
        n_mutate = max(2, seq_len // 5)
        n_alts_per_pos = 2

    if prediction is not None:
        positions = hybrid_position_selection(
            sample=sample,
            prediction=prediction,
            tokenizer=tokenizer,
            fixed_positions=fixed_positions,
            n_mutate=n_mutate,
            mutation_strategy=mutation_strategy,
            sequence_id=sequence_id,  
            timestep=int(timestep),
            timesteps=timesteps,
        )
    else:
        available_positions = [i for i in range(seq_len) if i not in fixed_positions]
        positions = np.random.choice(
            available_positions, min(n_mutate, len(available_positions)), replace=False
        ).tolist()

    mutation_seqs = []
    mutation_info = []

    for pos in positions:
        if pos in fixed_positions:
            continue

        original_aa = tokenizer.untokenize([sample[pos].item()])


        smart_candidates = mutation_strategy.get_beneficial_mutations(
            original_aa, sequence_id, n_select=n_alts_per_pos 
        )

        for alt_aa in smart_candidates:
            alt_id = safe_tokenize_single(tokenizer, alt_aa)
            mutated_sample = sample.clone()
            mutated_sample[pos] = alt_id

            mutation_seqs.append(tokenizer.untokenize(mutated_sample))
            mutation_info.append((pos, alt_id, base_score_smooth, original_aa, alt_aa))

    if mutation_seqs:
        mutation_scores = batch_evaluate_sequences(
            mutation_seqs,
            guidance_type,
            cache,
            target_protein,
            chain_id,
            model_runner,
            peptide_type,
        )

        for (pos, alt_id, baseline_score, orig_aa, alt_aa), mut_score in zip(
            mutation_info, mutation_scores
        ):
            if mut_score is not None:
                baseline_distance = abs(baseline_score - target_score)
                mutation_distance = abs(mut_score - target_score)

                if mutation_distance < baseline_distance:
                    reward = (baseline_distance - mutation_distance) / max(
                        baseline_distance, 0.1
                    )
                    grad = reward * dynamic_guidance_scale
                elif mut_score > baseline_score and mut_score > target_score:
                    improvement = mut_score - baseline_score
                    grad = improvement * dynamic_guidance_scale * 0.5
                else:

                    penalty = (mutation_distance - baseline_distance) / max(
                        mutation_distance, 0.1
                    )
                    grad = -penalty * dynamic_guidance_scale * 0.3

                grads[pos, alt_id] += grad / n_alts_per_pos


                improvement = (
                    1.0
                    if mutation_distance < baseline_distance
                    else (mut_score - baseline_score)
                )
                mutation_strategy.update_mutation_history(
                    orig_aa, alt_aa, improvement, sequence_id  
                )

    return grads, base_score_smooth



def batch_evaluate_sequences(
    sequences,
    guidance_type,
    cache,
    target_protein=None,
    chain_id=None,
    model_runner=None,
    peptide_type="cycle",
):
    results = []
    sequences_to_eval = []
    indices_to_eval = []

    for i, seq in enumerate(sequences):
        if seq in cache:
            results.append(cache.get(seq))
        else:
            sequences_to_eval.append(seq)
            indices_to_eval.append(i)
            results.append(None)

    if sequences_to_eval:
        if guidance_type == "structure":
            if peptide_type == "linear":
                structure_fn = linearpeptide_protein
            elif peptide_type == "disulfide":
                structure_fn = sspeptide_protein
            elif peptide_type == "cycle":
                structure_fn = cyclepeptide_protein
            else:
                raise ValueError(f"Unsupported peptide_type: {peptide_type}")

            preds = structure_fn(
                target_protein,
                sequences_to_eval,
                model_runner,
                chain_id,
                keep_temp=False,
            )
            for idx, (iptm, plddt, ipae) in zip(indices_to_eval, preds):
                score = iptm + plddt / 100 + (1 - ipae)
                cache.set(sequences[idx], score)
                results[idx] = score

        elif guidance_type == "permeability":
            for i, seq in zip(indices_to_eval, sequences_to_eval):
                score = permeable(seq)
                cache.set(seq, score)
                results[i] = score

    return results



def safe_tokenize_single(tokenizer, aa):

    result = tokenizer.tokenize(aa)
    if isinstance(result, (list, tuple)):
        return result[0]
    else:
        return result


class AdaptiveGuidanceScheduler:
    def __init__(
        self, base_scale=50.0, min_scale=30.0, max_scale=100.0, target_score=3.0
    ):
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_score = target_score
        self.score_history = deque(maxlen=15)
        self.best_score = float("-inf")
        self.stagnation_count = 0
        self.last_improvement_step = 0

    def get_guidance_scale(self, timestep, timesteps, current_score):
        progress = 1 - timestep / timesteps


        distance_to_target = abs(current_score - self.target_score)

        if current_score >= self.target_score:

            target_factor = 0.5 + 0.5 * np.exp(-distance_to_target)
        else:

            target_factor = 1.0 + distance_to_target * 0.5

        if progress > 0.8:
            if current_score >= self.target_score:
                time_factor = 0.8
            else:
                time_factor = 1.8
        elif progress > 0.6:
            time_factor = 1.2
        elif progress > 0.4:
            time_factor = 0.8
        else:
            time_factor = 0.5

        adaptive_factor = 1.0

        if current_score > self.best_score:
            self.best_score = current_score
            self.stagnation_count = 0
            self.last_improvement_step = timestep
            adaptive_factor = 0.9
        else:
            self.stagnation_count += 1
            stagnation_threshold = max(3, int(timesteps * 0.05))
            if self.stagnation_count > stagnation_threshold:
                stagnation_factor = min(2.0, 1.0 + self.stagnation_count * 0.1)
                adaptive_factor = stagnation_factor

        if len(self.score_history) >= 8:
            recent_scores = list(self.score_history)[-8:]
            recent_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

            if recent_trend <= 0:
                trend_factor = 1.0 + abs(recent_trend) * 10
                adaptive_factor *= min(trend_factor, 1.5)
            else:
                adaptive_factor *= 0.95

        if len(self.score_history) >= 10:
            score_var = np.var(list(self.score_history)[-10:])
            if score_var < 0.01:  
                adaptive_factor *= 1.3

        self.score_history.append(current_score)

        scale = self.base_scale * time_factor * adaptive_factor * target_factor
        return np.clip(scale, self.min_scale, self.max_scale)



def generate_d3pm(
    model,
    tokenizer,
    Q,
    Q_bar,
    timesteps,
    seq_len,
    batch_size=1,
    device="",
    guidance_scale=5.0,
    init_seqs=None,
    target_protein=None,
    chain_id=None,
    guidance_type="structure",
    fixed_positions=None,
    model_runner=None,
    target_score=None,
    peptide_type="cycle",
):

    valid_peptide_types = ["linear", "cycle", "disulfide"]
    if peptide_type not in valid_peptide_types:
        raise ValueError(
            f"peptide_type must be one of {valid_peptide_types}, got: {peptide_type}"
        )


    if target_score is None:
        if guidance_type == "structure":
            target_score = 3.0
        elif guidance_type == "permeability":
            target_score = -4.0
        else:
            target_score = 0.0

    if fixed_positions is None:
        fixed_positions = {}


    score_cache = ScoreCache(max_size=10000)
    aa_vocab = list("ACDEFGHIKLMNPQRSTVWY")
    mutation_strategy = IndependentMutationStrategy(aa_vocab, c=2.0)


    schedulers = [
        AdaptiveGuidanceScheduler(base_scale=guidance_scale, target_score=target_score)
        for _ in range(batch_size)
    ]


    best_tracker = IndividualBestTracker(batch_size)


    os.makedirs("./csvhighfold", exist_ok=True)
    pdb_id = (
        os.path.basename(target_protein).split(".")[0] if target_protein else "unknown"
    )
    

    csv_files = []
    csv_writers = []
    
    for seq_id in range(batch_size):
        csv_path = f"./csvhighfold/sequence_{seq_id}_{guidance_type}-{peptide_type}-{pdb_id}.csv"
        

        if os.path.exists(csv_path):
            os.remove(csv_path)
        
        csv_file = open(csv_path, "w")
        csv_file.write("step,sequence,score\n")
        csv_files.append(csv_file)
        # print(f"Created CSV file for sequence {seq_id}: {csv_path}")

    if init_seqs is not None:
        sample = torch.stack(
            [torch.tensor(tokenizer.tokenize(s), dtype=torch.long) for s in init_seqs]
        ).to(device)
        if sample.shape[0] == seq_len and sample.shape[1] == batch_size:
            sample = sample.T
    else:
        sample = torch.randint(
            0, tokenizer.K - 6, (batch_size, seq_len), dtype=torch.long, device=device
        )
        for i in range(batch_size):
            for pos, aa in fixed_positions.items():
                aa_token = safe_tokenize_single(tokenizer, aa)
                sample[i, pos] = aa_token


    Q, Q_bar = Q.to(device), Q_bar.to(device)
    reverse_steps = torch.arange(timesteps - 1, 0, -1, dtype=torch.long, device=device)
    guidance_steps = set(range(timesteps - 1, 0, -1))

    decay_rate = 0.9
    grad_accumulator = torch.zeros(
        (batch_size, seq_len, tokenizer.K - 6), device=device
    )
    base_score_smooth = [None for _ in range(batch_size)]
    score_momentum = 0.9

    all_scores = []
    all_sequences = []
    model.eval()


    target_sequence = extract_sequence_from_pdb(target_protein)
    target_tokenized = tokenizer.tokenize(target_sequence)
    target_len = len(target_tokenized)

    receptor = torch.zeros((batch_size, target_len), dtype=torch.long, device=device)
    receptor[:] = torch.tensor(target_tokenized, dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            for t in tqdm(
                reverse_steps, desc=f"Generating {peptide_type} peptide"
            ):
                timestep_tensor = torch.full(
                    (batch_size,), t, dtype=torch.long, device=device
                )

                prediction = model(sample, receptor, timestep_tensor)
                logits = prediction[:, :, : tokenizer.K]
                log_probs = F.log_softmax(logits, dim=-1).double()

                x_tminus1 = sample.clone()
                pad_id = tokenizer.vocab_size

                for i, s in enumerate(sample):
                    x_t_b = tokenizer.one_hot(s, pad_id)
                    A = torch.mm(x_t_b, Q[t].T)
                    Q_expand = (
                        Q_bar[t - 1].unsqueeze(0).expand(seq_len, tokenizer.K, tokenizer.K)
                    )
                    B_pred = log_probs[i].unsqueeze(2) * Q_expand
                    q_t = A.unsqueeze(1) * B_pred
                    p_theta_marg = torch.bmm(
                        q_t.transpose(1, 2), log_probs[i].unsqueeze(2)
                    ).squeeze()
                    p_theta_marg = p_theta_marg[:, : tokenizer.K - 6]
                    p_theta_marg = p_theta_marg / p_theta_marg.sum(dim=1, keepdim=True)

                    if t.item() in guidance_steps:
                        struct_grad, base_score_smooth[i] = (
                            adaptive_guidance_gradient_with_accumulation(
                                s,
                                tokenizer,
                                guidance_type,
                                score_cache,
                                target_protein,
                                chain_id,
                                model_runner,
                                mutation_strategy,
                                i, 
                                fixed_positions,
                                guidance_scale,
                                t.item(),
                                timesteps,
                                base_score_smooth[i],
                                score_momentum,
                                prediction=(
                                    prediction[i] if prediction.dim() == 3 else prediction
                                ),
                                scheduler=schedulers[i],  
                                target_score=target_score,
                                peptide_type=peptide_type,
                            )
                        )
                        grad_accumulator[i] = decay_rate * grad_accumulator[i] + struct_grad
                        guided_logits = (
                            torch.log(p_theta_marg + 1e-10) + grad_accumulator[i]
                        )
                        probs = F.softmax(guided_logits, dim=-1)
                    else:
                        probs = p_theta_marg

                    if t.item() == 1:
                        probs[:, tokenizer.K - 6 :] = 0
                        probs = probs / probs.sum(dim=1, keepdim=True)

                    for pos, aa in fixed_positions.items():
                        aa_id = safe_tokenize_single(tokenizer, aa)
                        probs[pos, :] = 0.0
                        probs[pos, aa_id] = 1.0

                    x_tminus1[i] = torch.multinomial(probs, num_samples=1).squeeze()
                    # x_tminus1[i] = torch.argmax(probs)

                sample = x_tminus1

                sequences = [tokenizer.untokenize(s) for s in sample]
                scores = batch_evaluate_sequences(
                    sequences,
                    guidance_type,
                    score_cache,
                    target_protein,
                    chain_id,
                    model_runner,
                    peptide_type,
                )

                best_tracker.update(t.item(), sequences, scores)

                for seq_id, (seq, score) in enumerate(zip(sequences, scores)):
                    if seq_id < len(csv_files):
                        csv_files[seq_id].write(f"{t.item()},{seq},{score:.4f}\n")
                        csv_files[seq_id].flush()  

                all_scores.append(scores)
                all_sequences.append(sequences)

    finally:
        for seq_id, csv_file in enumerate(csv_files):
            try:
                csv_file.close()
                print(f"Closed CSV file for sequence {seq_id}")
            except Exception as e:
                print(f"Error closing CSV file for sequence {seq_id}: {e}")

    final_sequences = [tokenizer.untokenize(s) for s in sample]
    final_scores = batch_evaluate_sequences(
        final_sequences,
        guidance_type,
        score_cache,
        target_protein,
        chain_id,
        model_runner,
        peptide_type,
    )


    best_tracker.update(0, final_sequences, final_scores)

    individual_results = best_tracker.get_results()

    print(f"Generation completed for {peptide_type} peptide (Independent Strategy).")
    print(f"Target score: {target_score}")
    print(f"Cache hit rate: {len(score_cache.cache)} entries")
    print(f"Created {batch_size} individual CSV files for each sequence")


    print("\n=== Independent Mutation Strategy Statistics ===")
    for seq_id in range(batch_size):
        print(mutation_strategy.get_sequence_stats_summary(seq_id))

    print(f"\n=== Individual Best Results (batch_size={batch_size}) ===")
    best_sequences = []
    for result in individual_results:
        seq_id = result["sequence_id"]
        best_seq = result["best_sequence"]
        best_score = result["best_score"]
        best_timestep = result["best_timestep"]

        target_achieved = "✓" if best_score >= target_score else "✗"
        print(f"Sequence {seq_id}: {best_seq}")
        print(
            f"  Best Score: {best_score:.3f} {target_achieved} (at timestep {best_timestep})"
        )
        print(f"  CSV saved to: ./csvhighfold/sequence_{seq_id}_{guidance_type}-{peptide_type}-{pdb_id}-independent.csv")
        best_sequences.append(best_seq)

    create_individual_visualization_plots(
        individual_results,
        tokenizer,
        guidance_type,
        target_score,
        f"{pdb_id}-independent",
        timesteps,
        batch_size,
    )

    return best_sequences


def create_individual_visualization_plots(
    individual_results,
    tokenizer,
    guidance_type,
    target_score,
    pdb_id,
    timesteps,
    batch_size,
):



    all_sequences = []
    for result in individual_results:
        all_sequences.extend(result["sequence_history"])

    all_tokens = []
    for s in all_sequences:
        try:
            tokens = tokenizer.tokenize(s)
            if isinstance(tokens, (list, tuple, torch.Tensor)):
                if isinstance(tokens, torch.Tensor):
                    all_tokens.append(tokens.long())
                else:
                    all_tokens.append(torch.tensor(tokens, dtype=torch.long))
            else:

                char_tokens = []
                for char in s:
                    char_token = tokenizer.tokenize(char)
                    if isinstance(char_token, (list, tuple)):
                        char_tokens.append(char_token[0])
                    else:
                        char_tokens.append(char_token)
                all_tokens.append(torch.tensor(np.array(char_tokens), dtype=torch.long))
        except Exception as e:
            print(f"Error tokenizing sequence {s}: {e}")
            char_tokens = []
            for char in s:
                try:
                    char_token = tokenizer.tokenize(char)
                    if isinstance(char_token, (list, tuple)):
                        char_tokens.append(char_token[0])
                    else:
                        char_tokens.append(char_token)
                except:
                    char_tokens.append(0) 
            all_tokens.append(torch.tensor(char_tokens, dtype=torch.long))

    if not all_tokens:
        print("Warning: No valid tokens for visualization")
        return

    max_len = max(len(tokens) for tokens in all_tokens)
    padded_tokens = []
    for tokens in all_tokens:
        if len(tokens) < max_len:
            padded = torch.zeros(max_len, dtype=torch.long)
            padded[: len(tokens)] = tokens
            padded_tokens.append(padded)
        else:
            padded_tokens.append(tokens[:max_len])

    token_array = torch.stack(padded_tokens)

    vocab = list("ACDEFGHIKLMNPQRSTVWY")
    vocab_size = len(vocab)
    seq_length = token_array.size(1)
    heatmap_matrix = torch.zeros((vocab_size, seq_length))

    for i in range(seq_length):
        for j, aa in enumerate(vocab):
            try:
                aa_id = tokenizer.tokenize(aa)
                if isinstance(aa_id, (list, tuple)):
                    aa_id = aa_id[0]
                aa_id = torch.tensor(aa_id, dtype=torch.long)

                matches = token_array[:, i] == aa_id
                heatmap_matrix[j, i] = matches.float().mean()
            except Exception as e:
                print(f"Error processing amino acid {aa} at position {i}: {e}")
                heatmap_matrix[j, i] = 0.0

    plt.figure(figsize=(max(12, seq_length * 0.8), 8))
    ax = sns.heatmap(
        heatmap_matrix.cpu().numpy(),
        cmap="viridis",
        cbar_kws={"shrink": 0.6},
        annot=False,
        fmt=".2f",
    )

    ax.set_xticks(np.arange(seq_length) + 0.5)
    ax.set_xticklabels(
        [f"Pos {i}" for i in range(seq_length)], rotation=45, fontsize=8, ha="right"
    )

    ax.set_yticks(np.arange(vocab_size) + 0.5)
    ax.set_yticklabels(vocab, rotation=0, fontsize=10)

    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Amino Acid", fontsize=12)
    plt.title(
        f"Amino Acid Distribution Heatmap ({guidance_type}, Target: {target_score}, Batch: {batch_size})",
        fontsize=14,
    )
    plt.tight_layout()

    heatmap_path = f"./piturehighfold/heatmap_{guidance_type}-{pdb_id}-target{target_score}-batch{batch_size}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {heatmap_path}")


    if batch_size == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    else:
        rows = (batch_size + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.flatten() if batch_size > 1 else [axes]
        else:
            axes = axes.flatten()

    for seq_id, result in enumerate(individual_results):
        if seq_id >= len(axes):
            break
        
        ax = axes[seq_id]
        score_history = result["score_history"]
        timestep_labels = list(range(len(score_history), 0, -1))

        ax.plot(timestep_labels, score_history, marker="o", linewidth=2, markersize=4)
        ax.axhline(
            y=target_score,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Target ({target_score})",
        )


        best_score = result["best_score"]
        best_timestep = result["best_timestep"]
        if best_timestep is not None:

            try:
                best_idx = len(score_history) - best_timestep
                if 0 <= best_idx < len(score_history):
                    ax.scatter(
                        best_timestep,
                        best_score,
                        color="red",
                        s=100,
                        zorder=5,
                        label=f"Best: {best_score:.3f}",
                    )
            except:
                pass

        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"{guidance_type.capitalize()} Score")
        ax.set_title(f"Sequence {seq_id} Score Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)


    for seq_id in range(len(individual_results), len(axes)):
        axes[seq_id].set_visible(False)

    plt.tight_layout()
    individual_plot_path = f"./piturehighfold/individual_scores_{guidance_type}-{pdb_id}-target{target_score}-batch{batch_size}.png"
    plt.savefig(individual_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved individual score plots to: {individual_plot_path}")

    plt.figure(figsize=(10, 6))
    final_scores = [result["best_score"] for result in individual_results]
    sequence_ids = [f"Seq {i}" for i in range(len(final_scores))]

    bars = plt.bar(sequence_ids, final_scores, alpha=0.7)
    plt.axhline(
        y=target_score,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Target Score ({target_score})",
    )

    for bar, score in zip(bars, final_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for i, score in enumerate(final_scores):
        if score >= target_score:
            bars[i].set_color("green")
            bars[i].set_alpha(0.8)

    plt.xlabel("Sequence ID")
    plt.ylabel(f"{guidance_type.capitalize()} Score")
    plt.title(f"Best Scores Comparison (Target: {target_score})")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    comparison_path = f"./score_comparison_{guidance_type}-{pdb_id}-target{target_score}-batch{batch_size}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved score comparison to: {comparison_path}")
