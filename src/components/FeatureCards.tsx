import { Fingerprint, Layers, ScanEye, UserRoundCheck } from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface Feature {
  icon: LucideIcon;
  title: string;
  description: string;
}

const FEATURES: Feature[] = [
  {
    icon: Fingerprint,
    title: "Personalized Matching",
    description:
      "Match scholarships to your academic and extracurricular profile.",
  },
  {
    icon: Layers,
    title: "AI-Powered Ranking",
    description:
      "Multiple machine-learning models work together to rank suitable scholarships.",
  },
  {
    icon: ScanEye,
    title: "Semantic Understanding",
    description:
      "SBERT captures meaningful similarity between student profiles and scholarship descriptions.",
  },
  { icon: UserRoundCheck, title: "Explainable Recommendations", description: "Understand which factors influenced each scholarship recommendation." },
];

export default function FeatureCards() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-12 max-w-xl">
        <p className="font-mono text-[0.68rem] uppercase tracking-[0.16em] text-[var(--color-brass)]">
          What powers it
        </p>
        <h2 className="mt-3 font-display text-[1.9rem] font-semibold tracking-tight text-[var(--color-navy)]">
          Built on a rigorous matching pipeline
        </h2>
      </div>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((feature) => (
          <div
            key={feature.title}
            className="rounded-2xl border border-[var(--color-line)] bg-white p-7 transition-shadow hover:shadow-[0_16px_40px_-24px_rgba(16,25,47,0.35)]"
          >
            <div className="mb-5 flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--color-blue-soft)]">
              <feature.icon size={20} className="text-[var(--color-blue)]" />
            </div>
            <h3 className="font-display text-lg font-semibold text-[var(--color-navy)]">
              {feature.title}
            </h3>
            <p className="mt-2.5 text-[0.9rem] leading-relaxed text-[var(--color-muted)]">
              {feature.description}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
