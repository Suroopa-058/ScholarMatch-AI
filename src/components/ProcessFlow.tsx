const STAGES = [
  { n: "01", label: "Student Profile" },
  { n: "02", label: "AI Matching" },
  { n: "03", label: "Ensemble Prediction" },
  { n: "04", label: "Scholarship Ranking" },
  { n: "05", label: "Explanation" },
];

export default function ProcessFlow() {
  return (
    <section id="how-it-works" className="bg-[var(--color-navy)] py-20">
      <div className="mx-auto max-w-6xl px-6">
        <div className="mb-14 max-w-xl">
          <p className="font-mono text-[0.68rem] uppercase tracking-[0.16em] text-[var(--color-brass)]">
            How it works
          </p>
          <h2 className="mt-3 font-display text-[1.9rem] font-semibold tracking-tight text-white">
            From profile to reasoned recommendation
          </h2>
        </div>

        <ol className="relative grid grid-cols-1 gap-x-4 gap-y-10 sm:grid-cols-5">
          <div
            className="absolute left-0 right-0 top-[13px] hidden h-px bg-white/15 sm:block"
            aria-hidden="true"
          />
          {STAGES.map((stage) => (
            <li key={stage.n} className="relative">
              <div className="mb-5 flex h-7 w-7 items-center justify-center rounded-full border border-[var(--color-brass)] bg-[var(--color-navy)] font-mono text-[0.62rem] text-[var(--color-brass)]">
                {stage.n}
              </div>
              <p className="font-display text-[0.98rem] font-medium text-white">
                {stage.label}
              </p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
