/** Base URLs for APIs (browser must reach published compose ports on the host). */
export const backendUrl =
  import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8081";
export const aiUrl = import.meta.env.VITE_AI_URL ?? "http://localhost:8000";
