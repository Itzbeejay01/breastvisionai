import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { apiHelpers } from "../api/client";
import { verdictLabel } from "../utils/verdict";
import { IMAGE_TYPES } from "../store/index";

const IMAGE_TYPE_OPTIONS = [
  { value: IMAGE_TYPES.RAW, label: "Raw (needs preprocessing)" },
  { value: IMAGE_TYPES.PROCESSED, label: "Pre-processed (normalized)" },
];

export default function ImageUploader({ onSelect, maxFiles = 1 }) {
  const [activeTab, setActiveTab] = useState("upload");
  const [imageType, setImageType] = useState(IMAGE_TYPES.RAW);
  const [gallery, setGallery] = useState([]);
  const [galleryLoading, setGalleryLoading] = useState(false);

  const onDrop = useCallback(
    (acceptedFiles) => {
      const files = acceptedFiles.slice(0, maxFiles);
      files.forEach((file) => {
        onSelect && onSelect({ file, imageType, preview: URL.createObjectURL(file) });
      });
    },
    [onSelect, maxFiles, imageType]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    fileRejections,
  } = useDropzone({
    onDrop,
    accept: {
      "image/png": [".png"],
      "image/jpeg": [".jpeg", ".jpg"],
      "application/dicom": [".dcm"],
    },
    multiple: maxFiles > 1,
  });

  useEffect(() => {
    if (activeTab !== "gallery") return;
    let cancelled = false;
    setGalleryLoading(true);
    (async () => {
      try {
        const data = await apiHelpers.getHistory(1, 50);
        if (!cancelled) setGallery(data.results || []);
      } catch (e) {
        if (!cancelled) setGallery([]);
      } finally {
        if (!cancelled) setGalleryLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [activeTab]);

  const selectFromGallery = async (item) => {
    const url = item.uploaded_image?.preview_url;
    if (!url) return;
    try {
      const blob = await (await fetch(url)).blob();
      const file = new File([blob], item.uploaded_image?.filename || "image", {
        type: blob.type,
      });
      onSelect && onSelect({
        file,
        imageType: item.uploaded_image?.image_type || IMAGE_TYPES.PROCESSED,
        preview: URL.createObjectURL(blob),
        isHistorical: true,
      });
    } catch (e) {
      console.error("Failed to fetch gallery image", e);
    }
  };

  return (
    <div className="image-uploader flex flex-col gap-4">
      <div className="bg-surface-container-low p-1 rounded-lg inline-flex w-fit shadow-[inset_0_1px_3px_rgba(0,0,0,0.05)]">
        {["upload", "gallery"].map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveTab(tab)}
            className={`px-6 py-2 rounded-md font-body-md text-body-md transition-colors touch-target ${
              activeTab === tab
                ? "bg-surface-container-lowest text-primary shadow-sm"
                : "text-on-surface-variant hover:bg-surface-container"
            }`}
          >
            {tab === "upload" ? "Upload" : "Gallery"}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4">
        <span className="text-sm text-gray-600">Image type:</span>
        {IMAGE_TYPE_OPTIONS.map((opt) => (
          <label key={opt.value} className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="imageType"
              checked={imageType === opt.value}
              onChange={() => setImageType(opt.value)}
              className="accent-malignant"
            />
            {opt.label}
          </label>
        ))}
      </div>

      {activeTab === "upload" && (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-malignant bg-red-50"
              : "border-gray-300 hover:border-gray-400"
          }`}
        >
          <input {...getInputProps()} />
          <span className="text-3xl mb-2 block">📁</span>
          {isDragActive ? (
            <p className="text-gray-700">Drop the image here …</p>
          ) : (
            <p className="text-gray-600">
              Drag & drop an image, or click to select (PNG, JPEG, DICOM)
            </p>
          )}
          {fileRejections.length > 0 && (
            <p className="text-xs text-malignant mt-2">
              Some files were rejected
            </p>
          )}
        </div>
      )}

      {activeTab === "gallery" && (
        <div className="overflow-y-auto max-h-72">
          {galleryLoading ? (
            <p className="text-gray-500">Loading gallery…</p>
          ) : gallery.length === 0 ? (
            <p className="text-gray-500">No previously processed images.</p>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 gap-3">
              {gallery.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => selectFromGallery(item)}
                  className="relative aspect-square rounded overflow-hidden border-2 border-transparent hover:border-malignant focus:border-malignant focus:outline-none"
                >
                  <img
                    src={item.uploaded_image?.preview_url}
                    alt={item.uploaded_image?.filename}
                    className="object-cover w-full h-full"
                  />
                  <span
                    className={`absolute bottom-0 left-0 right-0 text-[10px] px-1 py-0.5 ${
                      item.prediction_result === "malignant"
                        ? "bg-malignant"
                        : "bg-benign"
                    } text-white truncate`}
                  >
                    {verdictLabel(item.prediction_result)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
