import type { ApiError, AuthResponse, Explanation, RecommendationResponse, Scholarship, StudentProfile } from "../types/api";

const baseUrl = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${baseUrl}${path}`, { headers: { "Content-Type": "application/json", ...init?.headers }, ...init }); }
  catch { throw { message: "Unable to reach the backend. Check that it is running and try again." } satisfies ApiError; }
  if (!response.ok) { const body = await response.json().catch(() => null) as { detail?: ApiError; message?: string } | null; throw body?.detail ?? { message: body?.message ?? "Something went wrong. Please try again." } satisfies ApiError; }
  return response.json() as Promise<T>;
}
export const api = { recommend: (profile: StudentProfile) => request<RecommendationResponse>("/recommend", { method: "POST", body: JSON.stringify(profile) }), explain: (studentId: number, scholarshipId: string) => request<{ scholarship_id: string; explanation: Explanation[] }>("/explain", { method: "POST", body: JSON.stringify({ student_id: studentId, scholarship_id: scholarshipId }) }), scholarships: () => request<Scholarship[]>("/scholarships"), scholarship: (id: string) => request<Scholarship>(`/scholarships/${id}`), signup: (name: string, email: string, password: string) => request<AuthResponse>("/auth/signup", { method: "POST", body: JSON.stringify({ name, email, password }) }), login: (email: string, password: string) => request<AuthResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }) };
