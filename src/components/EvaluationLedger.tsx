import { CheckCircle2 } from "lucide-react";

const STEPS = [
  "Student profile received",
  "Features engineered",
  "Semantic similarity scored",
  "Ensemble models consulted",
  "Scholarships ranked",
];

/**
 * Signature hero visual — styled as an academic evaluation ledger,
 * echoing a transcript / certificate rather than a generic dashboard
 * mockup. Content here is illustrative of the pipeline stages only;
 * no scores or outcomes are implied.
 */
export default function EvaluationLedger() {
  return (
    <div className="relative">
      <div className="absolute -right-4 -top-4 h-full w-full rounded-2xl border border-[var(--color-brass)]/30 md:-right-6 md:-top-6" />
      <div className="relative rounded-2xl border border-[var(--color-line)] bg-[var(--color-paper-raised)] p-7 shadow-[0_20px_50px_-25px_rgba(16,25,47,0.35)] ledger-lines">
        <div className="mb-5 flex items-start justify-between">
          <div>
            <p className="font-mono text-[0.68rem] uppercase tracking-[0.16em] text-[var(--color-muted)]">
              Evaluation Ledger
            </p>
            <p className="font-display text-lg font-semibold text-[var(--color-navy)]">
              Matching Pipeline
            </p>
          </div>
          <div className="flex h-11 w-11 shrink-0 -rotate-6 items-center justify-center rounded-full border-2 border-[var(--color-brass)] font-mono text-[0.55rem] font-semibold uppercase tracking-wide text-[var(--color-brass)]">
            AI
          </div>
        </div>

        <ul className="space-y-[9px]">
          {STEPS.map((step, i) => (
            <li key={step} className="flex items-center gap-3 py-1.5">
              <CheckCircle2
                size={16}
                className="shrink-0 text-[var(--color-blue)]"
                style={{ opacity: 1 - i * 0.08 }}
              />
              <span className="text-[0.83rem] text-[var(--color-ink)]">{step}</span>
            </li>
          ))}
        </ul>

        <div className="mt-6 flex items-center justify-between border-t border-dashed border-[var(--color-line)] pt-4">
          <span className="font-mono text-[0.68rem] text-[var(--color-muted)]">
            SBERT · XGBoost · RF · LightGBM · CatBoost
          </span>
          <span className="rotate-[-4deg] rounded border border-[var(--color-brass)] px-2 py-0.5 font-mono text-[0.62rem] font-semibold uppercase tracking-wide text-[var(--color-brass)]">
            Explainable
          </span>
        </div>
      </div>
    </div>
  );
}
