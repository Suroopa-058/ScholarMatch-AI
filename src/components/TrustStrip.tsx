const STATS = [
  { value: "78,039", label: "Student records" },
  { value: "9,290", label: "Unique students" },
  { value: "10", label: "Scholarships modeled" },
  { value: "92,900", label: "Interactions learned from" },
];

export default function TrustStrip() {
  return (
    <section className="border-y border-[var(--color-line)] bg-white">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
          {STATS.map((stat) => (
            <div key={stat.label} className="text-center sm:text-left">
              <p className="font-mono text-2xl font-semibold text-[var(--color-navy)] sm:text-[1.7rem]">
                {stat.value}
              </p>
              <p className="mt-1 text-[0.78rem] text-[var(--color-muted)]">
                {stat.label}
              </p>
            </div>
          ))}
        </div>
        <p className="mt-6 text-center text-[0.72rem] text-[var(--color-muted)] sm:text-left">
          Dataset scale from the current training pipeline. Scholarship
          metadata used in this prototype is illustrative and not verified
          government or international listings.
        </p>
      </div>
    </section>
  );
}
