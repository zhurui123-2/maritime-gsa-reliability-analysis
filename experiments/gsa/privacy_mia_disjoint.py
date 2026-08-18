"""Population-disjoint black-box/white-box MIA with a privacy-utility sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import load_complete_entities, prepare_data
    from experiments.gsa.federated import FederatedTrainer
    from experiments.gsa.metrics import evaluate_all_pairs
    from experiments.gsa.privacy_mia import bootstrap_auc_ci
else:
    from .data import load_complete_entities, prepare_data
    from .federated import FederatedTrainer
    from .metrics import evaluate_all_pairs
    from .privacy_mia import bootstrap_auc_ci


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ("ais", "weather", "registry", "emissions")
PAIR_NAMES = tuple(
    f"{left}_{right}_cosine"
    for left_index, left in enumerate(MODALITIES)
    for right in MODALITIES[left_index + 1 :]
)
BLACKBOX_FEATURES = [*PAIR_NAMES, "score_mean", "score_std", "score_entropy"]
WHITEBOX_FEATURES = ["alignment_loss", "score_entropy", "gradient_norm"]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def id_hash(values: set[str]) -> str:
    payload = json.dumps(sorted(values), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_populations(frame: pd.DataFrame, seed: int, target_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    indices = np.arange(len(frame))
    auxiliary_index, target_index = train_test_split(
        indices,
        test_size=target_fraction,
        random_state=seed,
        stratify=frame["federated_node"],
    )
    auxiliary = frame.iloc[np.sort(auxiliary_index)].reset_index(drop=True)
    target = frame.iloc[np.sort(target_index)].reset_index(drop=True)
    auxiliary_ids = set(auxiliary["MMSI"].astype(str))
    target_ids = set(target["MMSI"].astype(str))
    overlap = auxiliary_ids & target_ids
    if overlap:
        raise RuntimeError(f"Population leakage: {len(overlap)} MMSIs occur in both populations.")
    manifest = {
        "population_seed": seed,
        "target_fraction": target_fraction,
        "auxiliary_entities": len(auxiliary_ids),
        "target_entities": len(target_ids),
        "population_overlap": 0,
        "auxiliary_mmsi_sha256": id_hash(auxiliary_ids),
        "target_mmsi_sha256": id_hash(target_ids),
        "auxiliary_nodes": auxiliary["federated_node"].value_counts().sort_index().to_dict(),
        "target_nodes": target["federated_node"].value_counts().sort_index().to_dict(),
    }
    return auxiliary, target, manifest


def balanced_positions(trainer: FederatedTrainer, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    members = trainer.data.splits.train
    nonmembers = trainer.data.splits.test
    size = min(len(members), len(nonmembers))
    member_sample = rng.choice(members, size=size, replace=False)
    nonmember_sample = rng.choice(nonmembers, size=size, replace=False)
    positions = np.concatenate([member_sample, nonmember_sample])
    labels = np.concatenate([np.ones(size, dtype=int), np.zeros(size, dtype=int)])
    return positions, labels


def blackbox_features(
    trainer: FederatedTrainer,
    positions: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    trainer.model.eval()
    index = torch.as_tensor(positions, dtype=torch.long, device=trainer.device)
    with torch.no_grad():
        embeddings = trainer.model({name: trainer.features[name][index] for name in MODALITIES})
        rng = np.random.default_rng(seed)
        if noise_std > 0:
            embeddings = {
                name: F.normalize(
                    value
                    + torch.as_tensor(
                        rng.normal(0.0, noise_std, size=tuple(value.shape)),
                        dtype=value.dtype,
                        device=value.device,
                    ),
                    dim=-1,
                )
                for name, value in embeddings.items()
            }
        pair_scores = torch.stack(
            [
                F.cosine_similarity(embeddings[left], embeddings[right], dim=-1)
                for left_index, left in enumerate(MODALITIES)
                for right in MODALITIES[left_index + 1 :]
            ],
            dim=1,
        )
        probabilities = torch.softmax(pair_scores, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        result = torch.cat(
            [
                pair_scores,
                pair_scores.mean(dim=1, keepdim=True),
                pair_scores.std(dim=1, keepdim=True),
                entropy.unsqueeze(1),
            ],
            dim=1,
        )
    trainer.model.train()
    return result.cpu().numpy().astype(np.float32)


def whitebox_features(trainer: FederatedTrainer, positions: np.ndarray) -> np.ndarray:
    original = trainer.membership_features(positions)
    # Original columns: loss, confidence, entropy, embedding_norm, gradient_norm.
    return original[:, [0, 2, 4]].astype(np.float32)


def noisy_retrieval_mrr(trainer: FederatedTrainer, noise_std: float, seed: int) -> float:
    clean = trainer.embeddings(trainer.data.splits.test)
    rng = np.random.default_rng(seed)
    noisy = {}
    for name, values in clean.items():
        perturbed = values + rng.normal(0.0, noise_std, size=values.shape).astype(np.float32)
        norm = np.linalg.norm(perturbed, axis=1, keepdims=True).clip(min=1e-12)
        noisy[name] = perturbed / norm
    rows = evaluate_all_pairs(noisy)
    return float(np.mean([float(row["mrr"]) for row in rows]))


def fit_attack(x: np.ndarray, y: np.ndarray):
    attack = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=20260725),
    )
    attack.fit(x, y)
    return attack


def cache_key(noise_std: float) -> str:
    return f"black_{noise_std:.6f}".replace(".", "p")


def collect_model_features(
    trainer: FederatedTrainer,
    rng_seed: int,
    noise_levels: list[float],
) -> dict[str, np.ndarray]:
    positions, labels = balanced_positions(trainer, np.random.default_rng(rng_seed))
    payload = {"membership": labels}
    for noise_index, noise_std in enumerate(noise_levels):
        payload[cache_key(noise_std)] = blackbox_features(
            trainer, positions, noise_std, rng_seed + 100_000 * noise_index
        )
    payload["white"] = whitebox_features(trainer, positions)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--ablations", default="A,D")
    parser.add_argument("--shadow-models", type=int, default=50)
    parser.add_argument("--target-seeds", default="910001,910002")
    parser.add_argument("--population-seed", type=int, default=20260725)
    parser.add_argument("--target-population-fraction", type=float, default=0.50)
    parser.add_argument("--embedding-noise-stds", default="0,0.01,0.02,0.05,0.10")
    parser.add_argument(
        "--classification-weight-override",
        type=float,
        default=None,
        help="Optional privacy audit of reduced identity-classification memorization.",
    )
    parser.add_argument("--output-name", default="mia_population_disjoint")
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.classification_weight_override is not None:
        config["classification_weight"] = float(args.classification_weight_override)
    frame = load_complete_entities(PROJECT_ROOT / config["input"])
    auxiliary, target_population, population_manifest = split_populations(
        frame, args.population_seed, args.target_population_fraction
    )
    if min(len(auxiliary), len(target_population)) < 500:
        raise ValueError("Each disjoint population must contain at least 500 entities.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    ablations = parse_csv(args.ablations)
    target_seeds = parse_int_csv(args.target_seeds)
    noise_levels = parse_float_csv(args.embedding_noise_stds)
    output = PROJECT_ROOT / config["output_dir"] / args.output_name
    cache = output / "shadow_cache"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    (output / "population_manifest.json").write_text(
        json.dumps(population_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_rows: list[dict[str, object]] = []
    utility_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for ablation in ablations:
        shadow_payloads = []
        for shadow_id in range(args.shadow_models):
            seed = 200_000 + shadow_id
            path = cache / f"{ablation}_shadow_{shadow_id:03d}.npz"
            if path.exists():
                payload = dict(np.load(path, allow_pickle=False))
                print(f"ablation={ablation} shadow={shadow_id + 1}/{args.shadow_models} resumed", flush=True)
            else:
                print(f"ablation={ablation} shadow={shadow_id + 1}/{args.shadow_models} training", flush=True)
                prepared = prepare_data(
                    auxiliary,
                    seed,
                    float(config["test_fraction"]),
                    float(config["validation_fraction"]),
                )
                shadow = FederatedTrainer(prepared, config, ablation, seed, device)
                shadow.train()
                payload = collect_model_features(shadow, seed, noise_levels)
                np.savez_compressed(path, **payload)
            shadow_payloads.append(payload)

        shadow_y = np.concatenate([payload["membership"] for payload in shadow_payloads])
        attacks: dict[tuple[str, float], object] = {}
        sanity_attacks: dict[tuple[str, float], object] = {}
        permuted_shadow_y = np.random.default_rng(args.population_seed).permutation(shadow_y)
        for noise_std in noise_levels:
            key = cache_key(noise_std)
            shadow_x = np.concatenate([payload[key] for payload in shadow_payloads])
            attacks[("black_box", noise_std)] = fit_attack(shadow_x, shadow_y)
            sanity_attacks[("black_box", noise_std)] = fit_attack(shadow_x, permuted_shadow_y)
        white_shadow_x = np.concatenate([payload["white"] for payload in shadow_payloads])
        attacks[("white_box", 0.0)] = fit_attack(white_shadow_x, shadow_y)
        sanity_attacks[("white_box", 0.0)] = fit_attack(white_shadow_x, permuted_shadow_y)

        for target_seed in target_seeds:
            print(f"ablation={ablation} target_seed={target_seed} training", flush=True)
            prepared = prepare_data(
                target_population,
                target_seed,
                float(config["test_fraction"]),
                float(config["validation_fraction"]),
            )
            auxiliary_ids = set(auxiliary["MMSI"].astype(str))
            target_ids = set(prepared.entity_ids.astype(str))
            if auxiliary_ids & target_ids:
                raise RuntimeError("Auxiliary/target population overlap detected after preprocessing.")
            target = FederatedTrainer(prepared, config, ablation, target_seed, device)
            artifacts = target.train()
            payload = collect_model_features(target, target_seed, noise_levels)
            y_test = payload["membership"]

            for noise_std in noise_levels:
                x_test = payload[cache_key(noise_std)]
                probability = attacks[("black_box", noise_std)].predict_proba(x_test)[:, 1]
                sanity_probability = sanity_attacks[("black_box", noise_std)].predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, probability))
                ci_low, ci_high = bootstrap_auc_ci(
                    y_test, probability, target_seed + int(noise_std * 1_000_000), repeats=2000
                )
                retrieval_mrr = noisy_retrieval_mrr(
                    target, noise_std, target_seed + int(noise_std * 1_000_000)
                )
                summary_rows.append({
                    "ablation": ablation,
                    "target_seed": target_seed,
                    "threat_model": "black_box",
                    "embedding_noise_std": noise_std,
                    "shadow_models": args.shadow_models,
                    "attack_auc": auc,
                    "attack_auc_ci95_low": ci_low,
                    "attack_auc_ci95_high": ci_high,
                    "attack_accuracy": float(accuracy_score(y_test, probability >= 0.5)),
                    "permuted_label_sanity_auc": float(roc_auc_score(y_test, sanity_probability)),
                    "retrieval_mean_mrr": retrieval_mrr,
                })
                utility_rows.append({
                    "ablation": ablation,
                    "target_seed": target_seed,
                    "embedding_noise_std": noise_std,
                    "retrieval_mean_mrr": retrieval_mrr,
                })
                print(
                    f"ablation={ablation} target={target_seed} black noise={noise_std:.3f} "
                    f"auc={auc:.6f} mrr={retrieval_mrr:.6f}",
                    flush=True,
                )

            white_probability = attacks[("white_box", 0.0)].predict_proba(payload["white"])[:, 1]
            white_sanity_probability = sanity_attacks[("white_box", 0.0)].predict_proba(payload["white"])[:, 1]
            white_auc = float(roc_auc_score(y_test, white_probability))
            ci_low, ci_high = bootstrap_auc_ci(y_test, white_probability, target_seed + 700_000, repeats=2000)
            summary_rows.append({
                "ablation": ablation,
                "target_seed": target_seed,
                "threat_model": "white_box",
                "embedding_noise_std": 0.0,
                "shadow_models": args.shadow_models,
                "attack_auc": white_auc,
                "attack_auc_ci95_low": ci_low,
                "attack_auc_ci95_high": ci_high,
                "attack_accuracy": float(accuracy_score(y_test, white_probability >= 0.5)),
                "permuted_label_sanity_auc": float(roc_auc_score(y_test, white_sanity_probability)),
                "retrieval_mean_mrr": noisy_retrieval_mrr(target, 0.0, target_seed),
            })
            split_rows.append({
                "ablation": ablation,
                "target_seed": target_seed,
                "auxiliary_target_overlap": 0,
                "target_split_hash": prepared.split_hash,
                "target_train_size": len(prepared.splits.train),
                "target_validation_size": len(prepared.splits.validation),
                "target_test_size": len(prepared.splits.test),
                "best_round": int(artifacts.checkpoint["best_round"]),
                "best_validation_mean_mrr": float(artifacts.checkpoint["best_validation_mean_mrr"]),
            })
            pd.DataFrame(summary_rows).to_csv(
                output / "mia_v2_results.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(split_rows).to_csv(
                output / "split_audit.csv", index=False, encoding="utf-8-sig"
            )

    results = pd.DataFrame(summary_rows)
    results.to_csv(output / "mia_v2_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(utility_rows).to_csv(
        output / "privacy_utility_curve.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(split_rows).to_csv(output / "split_audit.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "protocol_version": 2,
                "population_disjoint": True,
                "black_box_features": BLACKBOX_FEATURES,
                "white_box_features": WHITEBOX_FEATURES,
                "embedding_noise_stds": noise_levels,
                "embedding_noise_claim": "empirical black-box output perturbation; not differential privacy",
                "classification_weight": float(config["classification_weight"]),
                "classification_weight_override": args.classification_weight_override,
                "ablations": ablations,
                "shadow_models": args.shadow_models,
                "target_seeds": target_seeds,
                **population_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
