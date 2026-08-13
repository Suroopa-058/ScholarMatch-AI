import type { AuthResponse } from "../types/api";
const AUTH_KEY = "scholarmatch.auth";
export function getAuth(): AuthResponse | null { try { const raw = localStorage.getItem(AUTH_KEY); return raw ? JSON.parse(raw) as AuthResponse : null; } catch { return null; } }
export function saveAuth(value: AuthResponse) { localStorage.setItem(AUTH_KEY, JSON.stringify(value)); }
export function clearAuth() { localStorage.removeItem(AUTH_KEY); }
