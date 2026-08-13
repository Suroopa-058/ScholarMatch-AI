import type { StudentProfile } from "../types/api";
export const PROFILE_KEY = "scholarmatch.profile";
export const RESULTS_KEY = "scholarmatch.results";
export function loadProfile(): StudentProfile | null { try { const raw = sessionStorage.getItem(PROFILE_KEY); return raw ? JSON.parse(raw) as StudentProfile : null; } catch { return null; } }
export function saveProfile(profile: StudentProfile) { sessionStorage.setItem(PROFILE_KEY, JSON.stringify(profile)); }
export function percent(value: number) { return `${Math.round(value * 100)}%`; }
