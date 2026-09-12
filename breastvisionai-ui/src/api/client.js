import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "/api",
  timeout: 60000,
  withCredentials: true,
  xsrfCookieName: "csrftoken",
  xsrfHeaderName: "X-CSRFToken",
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (!err.response) {
      err.isNetworkError = true;
    }
    return Promise.reject(err);
  }
);

async function requestWithRetry(config, retries = 3, delay = 500) {
  for (let i = 0; i < retries; i++) {
    try {
      return await api.request(config);
    } catch (e) {
      const shouldRetry = !e.response || e.response.status >= 500 || e.isNetworkError;
      if (!shouldRetry || i === retries - 1) throw e;
      await new Promise((r) => setTimeout(r, delay * 2 ** i));
    }
  }
  throw new Error("Request failed after retries");
}

export const apiHelpers = {
  getCsrf: async () => {
    const res = await requestWithRetry({ url: "/auth/csrf/", method: "get" });
    return res.data;
  },

  login: async (username, password) => {
    await apiHelpers.getCsrf();
    const res = await requestWithRetry({
      url: "/auth/login/",
      method: "post",
      data: { username, password },
    });
    return res.data;
  },

  logout: async () => {
    await apiHelpers.getCsrf();
    const res = await requestWithRetry({ url: "/auth/logout/", method: "post" });
    return res.data;
  },

  currentUser: async () => {
    const res = await requestWithRetry({ url: "/auth/me/", method: "get" });
    return res.data;
  },

  predictImage: async (file, imageType = "raw") => {
    const form = new FormData();
    form.append("image", file);
    form.append("image_type", imageType);
    const res = await requestWithRetry({
      url: "/predict/",
      method: "post",
      data: form,
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  predictBatch: async (files, imageType = "raw") => {
    const form = new FormData();
    files.forEach((f) => form.append("images", f));
    form.append("image_type", imageType);
    const res = await requestWithRetry({
      url: "/predict/batch/",
      method: "post",
      data: form,
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  getModels: async () => {
    const res = await requestWithRetry({ url: "/models/", method: "get" });
    return res.data;
  },

  getEnsemble: async () => {
    const res = await requestWithRetry({ url: "/ensemble/", method: "get" });
    return res.data;
  },

  getHistory: async (page = 1, limit = 20) => {
    const res = await requestWithRetry({
      url: "/history/",
      method: "get",
      params: { page, limit },
    });
    return res.data;
  },

  getHistoryDetail: async (id) => {
    const res = await requestWithRetry({ url: `/history/${id}/`, method: "get" });
    return res.data;
  },

  uploadImage: async (file, imageType = "raw") => {
    const form = new FormData();
    form.append("image", file);
    form.append("image_type", imageType);
    const res = await requestWithRetry({
      url: "/upload/",
      method: "post",
      data: form,
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  reportUrl: (id) => {
    const base = api.defaults.baseURL.replace(/\/$/, "");
    return `${base}/report/${id}/`;
  },
};

export default api;
