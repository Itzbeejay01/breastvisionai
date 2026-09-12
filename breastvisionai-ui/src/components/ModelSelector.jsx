export default function ModelSelector({ modelBreakdown, psoWeights, fusionWeights }) {
  const breakdown = modelBreakdown || {};
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    return <p className="text-sm text-gray-500">No model breakdown available.</p>;
  }

  const maxProb = Math.max(...entries.map(([, v]) => v.probability), 0.001);

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">Model Breakdown</h4>
      <div className="space-y-2">
        {entries.map(([name, info]) => {
          const prob = info.probability ?? 0;
          const weight = info.weight ?? psoWeights?.[name] ?? fusionWeights?.[name] ?? 0;
          const barWidth = (prob / maxProb) * 100;
          return (
            <div key={name}>
              <div className="flex justify-between text-xs text-gray-700">
                <span>{name}</span>
                <span>
                  prob {prob.toFixed(3)} · weight {weight.toFixed(3)}
                </span>
              </div>
              <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-sky-500 rounded-full transition-all"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {psoWeights && (
        <div className="pt-2 text-xs text-gray-500">
          PSO weights:{" "}
          {Object.entries(psoWeights)
            .map(([k, v]) => `${k}: ${v.toFixed(3)}`)
            .join(", ")}
        </div>
      )}
    </div>
  );
}
