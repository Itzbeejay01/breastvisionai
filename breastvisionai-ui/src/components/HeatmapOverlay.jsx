export default function HeatmapOverlay({ originalSrc, heatmapBase64 }) {
  if (!heatmapBase64) {
    return (
      <div className="text-sm text-gray-500">
        No heatmap available for this image.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-gray-700">
        Grad-CAM Heatmap
      </h4>
      <div className="relative inline-block rounded overflow-hidden border border-gray-200">
        {originalSrc && (
          <img
            src={originalSrc}
            alt="Original"
            className="max-w-full h-auto block"
          />
        )}
        <img
          src={heatmapBase64}
          alt="Grad-CAM heatmap overlay"
          className="absolute top-0 left-0 w-full h-full object-cover mix-blend-overlay opacity-70"
        />
      </div>
    </div>
  );
}
