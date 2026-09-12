import { useEffect, useState } from "react";
import { apiHelpers } from "../api/client";

export default function SystemStatus() {
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState(null);
  const [models, setModels] = useState([]);
  const [healthy, setHealthy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [ensemble, modelList] = await Promise.all([
          apiHelpers.getEnsemble(),
          apiHelpers.getModels(),
        ]);
        if (!cancelled) {
          setConfig(ensemble);
          setModels(modelList);
          setHealthy(true);
        }
      } catch (e) {
        if (!cancelled) setHealthy(false);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = models.filter((m) => m.selected_by_pso);

  if (loading) {
    return (
      <div className="p-4 text-center text-on-surface-variant text-body-sm">
        Checking system status…
      </div>
    );
  }

  return (
    <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <StatusCard
        label="API Status"
        value={healthy ? "Online" : "Offline"}
        color={healthy ? "text-secondary" : "text-error"}
        pulse={healthy}
      />
      <StatusCard label="Models Loaded" value={String(models.length)} />
      <StatusCard label="PSO-Selected" value={String(selected.length)} />
      <StatusCard
        label="Ensemble"
        value={config?.meta_learner?.type || "GradientBoosting"}
      />
    </section>
  );
}

function StatusCard({ label, value, color, pulse }) {
  return (
    <div className="bg-surface-container-lowest p-4 rounded-lg shadow-card border border-surface-container-high flex items-center justify-between">
      <div>
        <p className="font-label-caps text-label-caps text-on-surface-variant mb-1">
          {label}
        </p>
        <p className={`font-data-num text-data-num ${color || "text-on-surface"}`}>
          {value}
        </p>
      </div>
      {pulse !== undefined && (
        <span
          className={`w-3 h-3 rounded-full ${
            pulse ? "bg-secondary pulse-dot" : "bg-error"
          }`}
        />
      )}
    </div>
  );
}
