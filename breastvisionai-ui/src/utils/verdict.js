/**
 * Human-readable verdict label for the UI.
 * Internal value stays "benign"; we display "NON-MALIGNANT".
 */
export function verdictLabel(value) {
  if (value === "benign") return "NON-MALIGNANT";
  return String(value || "n/a").toUpperCase();
}
