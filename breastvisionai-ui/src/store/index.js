import { create } from "zustand";
import { apiHelpers } from "../api/client";

export const IMAGE_TYPES = {
  RAW: "raw",
  PROCESSED: "processed",
};

const useStore = create((set, _get) => ({
  user: null,
  authStatus: "checking",
  history: [],
  historyMeta: { count: 0, next: null, previous: null },
  modelConfig: null,
  currentResult: null,
  status: "idle",
  error: null,

  setResult: (result) => set({ currentResult: result }),
  clearResult: () => set({ currentResult: null }),
  setError: (error) => set({ error }),

  initializeAuth: async () => {
    try {
      const user = await apiHelpers.currentUser();
      set({ user, authStatus: "authenticated" });
    } catch {
      set({ user: null, authStatus: "anonymous" });
    }
  },

  login: async (username, password) => {
    const user = await apiHelpers.login(username, password);
    set({ user, authStatus: "authenticated" });
    return user;
  },

  logout: async () => {
    await apiHelpers.logout();
    set({ user: null, authStatus: "anonymous" });
  },

  fetchHistoryDetail: async (id) => {
    set({ status: "loading", error: null });
    try {
      const data = await apiHelpers.getHistoryDetail(id);
      set({ currentResult: data, status: "idle" });
      return data;
    } catch (e) {
      set({ status: "error", error: e.message || "Failed to load result" });
      throw e;
    }
  },

  fetchHistory: async (page = 1) => {
    set({ status: "loading", error: null });
    try {
      const data = await apiHelpers.getHistory(page);
      set({
        history: data.results || [],
        historyMeta: { count: data.count, next: data.next, previous: data.previous },
        status: "idle",
      });
    } catch (e) {
      set({ status: "error", error: e.message || "Failed to load history" });
    }
  },

  fetchEnsemble: async () => {
    set({ status: "loading", error: null });
    try {
      const [ensemble, models] = await Promise.all([
        apiHelpers.getEnsemble(),
        apiHelpers.getModels(),
      ]);
      set({ modelConfig: { ensemble, models }, status: "idle" });
    } catch (e) {
      set({ status: "error", error: e.message || "Failed to load ensemble" });
    }
  },

  predict: async (file, imageType = IMAGE_TYPES.RAW) => {
    set({ status: "loading", error: null });
    try {
      const result = await apiHelpers.predictImage(file, imageType);
      set({ currentResult: result, status: "idle" });
      return result;
    } catch (e) {
      set({ status: "error", error: e.message || "Prediction failed" });
      throw e;
    }
  },

  predictBatch: async (files, imageType = IMAGE_TYPES.RAW) => {
    set({ status: "loading", error: null });
    try {
      const result = await apiHelpers.predictBatch(files, imageType);
      set({ status: "idle" });
      return result;
    } catch (e) {
      set({ status: "error", error: e.message || "Batch prediction failed" });
      throw e;
    }
  },
}));

export default useStore;
