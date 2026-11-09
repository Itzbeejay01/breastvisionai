import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DiagnosisPage from "./DiagnosisPage";

const mocks = vi.hoisted(() => ({
  state: {},
  routeId: undefined,
  explain: vi.fn(),
  fetchDetail: vi.fn(),
  setState: vi.fn(),
}));

vi.mock("react-router-dom", () => ({
  useParams: () => ({ id: mocks.routeId }),
  useLocation: () => ({ state: null }),
}));

vi.mock("../store/index", () => {
  const store = () => mocks.state;
  store.getState = () => mocks.state;
  store.setState = (update) => {
    mocks.state = { ...mocks.state, ...(typeof update === "function" ? update(mocks.state) : update) };
    mocks.setState(update);
  };
  return { default: store };
});

vi.mock("../components/ImageUploader", () => ({ default: () => <div>Image uploader</div> }));
vi.mock("../components/ModelSelector", () => ({ default: () => null }));
vi.mock("../components/ReportGenerator", () => ({ default: () => null }));
vi.mock("../components/ExplanationPanel", () => ({
  default: ({ result }) => result?.explanation_data
    ? <div role="region" aria-label="Detailed explanation">Detailed image evidence</div>
    : <div role="region" aria-label="Detailed explanation">No details</div>,
}));

const prediction = {
  id: 17,
  prediction_result: "benign",
  prediction_probability: 0.2,
  confidence: 0.8,
  fused_probability: 0.3,
  model_breakdown: {},
  uploaded_image: { filename: "scan.png", preview_url: "/scan.png" },
  timestamp: "2026-11-09T00:00:00Z",
  meta_feature_schema: ["EfficientNet", "ResNet", "VGG16", "Late Fusion"],
};

function renderWithResult() {
  const view = render(<DiagnosisPage />);
  act(() => {
    mocks.state.currentResult = prediction;
  });
  view.rerender(<DiagnosisPage />);
  return view;
}

describe("DiagnosisPage on-demand explanations", () => {
  beforeEach(() => {
    mocks.routeId = undefined;
    mocks.setState.mockClear();
    mocks.fetchDetail.mockReset();
    mocks.explain.mockReset();
    mocks.state = {
      currentResult: null,
      predict: vi.fn(),
      fetchHistoryDetail: mocks.fetchDetail,
      explainPrediction: mocks.explain,
    };
  });

  it("generates only after click, hides, then reveals cached data without regenerating", async () => {
    mocks.explain.mockImplementation(async () => {
      mocks.state.currentResult = {
        ...mocks.state.currentResult,
        explanation_data: { version: "cached" },
      };
    });
    renderWithResult();

    expect(screen.getByText("How Does the AI Reach This Result?")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Detailed explanation" })).toBeNull();
    expect(mocks.explain).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    expect(await screen.findByRole("region", { name: "Detailed explanation" })).toBeTruthy();
    expect(mocks.explain).toHaveBeenCalledTimes(1);
    expect(mocks.explain).toHaveBeenCalledWith(17);

    fireEvent.click(screen.getByRole("button", { name: "Hide Explanation" }));
    expect(screen.queryByRole("region", { name: "Detailed explanation" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    expect(await screen.findByRole("region", { name: "Detailed explanation" })).toBeTruthy();
    expect(mocks.explain).toHaveBeenCalledTimes(1);
  });

  it("shows a retry after failure and resets explanation state for another record", async () => {
    mocks.explain
      .mockRejectedValueOnce(new Error("temporary error"))
      .mockImplementationOnce(async () => {
        mocks.state.currentResult = {
          ...mocks.state.currentResult,
          explanation_data: { version: "loaded" },
        };
      });
    const view = renderWithResult();
    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    expect((await screen.findByRole("alert")).textContent).toContain("temporary error");
    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    expect(await screen.findByRole("region", { name: "Detailed explanation" })).toBeTruthy();

    mocks.routeId = "22";
    mocks.fetchDetail.mockImplementation(async () => ({
      ...prediction,
      id: 22,
      explanation_data: {},
    }));
    view.rerender(<DiagnosisPage />);
    await waitFor(() => expect(mocks.fetchDetail).toHaveBeenCalledWith("22"));
    await waitFor(() => expect(mocks.state.currentResult?.id).toBe(22));
    view.rerender(<DiagnosisPage />);
    expect(screen.queryByRole("region", { name: "Detailed explanation" })).toBeNull();
    expect(screen.getByRole("button", { name: "Explain This Result" })).toBeTruthy();
  });

  it("shows progress and disables the action while detail is generating", async () => {
    let finish;
    mocks.explain.mockImplementation(() => new Promise((resolve) => {
      finish = () => {
        mocks.state.currentResult = {
          ...mocks.state.currentResult,
          explanation_data: { version: "done" },
        };
        resolve();
      };
    }));
    renderWithResult();
    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    const progress = await screen.findByRole("button", { name: "Generating explanation..." });
    expect(progress.disabled).toBe(true);
    await act(async () => finish());
    expect(await screen.findByRole("region", { name: "Detailed explanation" })).toBeTruthy();
  });

  it("does not reveal an old request after opening a different history record", async () => {
    let finish;
    mocks.explain.mockImplementation(() => new Promise((resolve) => {
      finish = () => {
        if (String(mocks.state.currentResult?.id) === "17") {
          mocks.state.currentResult = {
            ...mocks.state.currentResult,
            explanation_data: { version: "old-record" },
          };
        }
        resolve();
      };
    }));
    mocks.fetchDetail.mockResolvedValue({
      ...prediction,
      id: 22,
      explanation_data: {},
    });
    const view = renderWithResult();
    fireEvent.click(screen.getByRole("button", { name: "Explain This Result" }));
    expect(await screen.findByRole("button", { name: "Generating explanation..." })).toBeTruthy();

    mocks.routeId = "22";
    view.rerender(<DiagnosisPage />);
    await waitFor(() => expect(mocks.state.currentResult?.id).toBe(22));
    view.rerender(<DiagnosisPage />);
    await act(async () => finish());
    expect(screen.queryByRole("region", { name: "Detailed explanation" })).toBeNull();
    expect(screen.getByRole("button", { name: "Explain This Result" })).toBeTruthy();
  });
});
