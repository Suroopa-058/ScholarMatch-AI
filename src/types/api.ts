export interface StudentProfile { student_id: number; semester: number; gpa: number; extracurricular_point: number; total_credits: number; class: string; has_failed_course: boolean; student_year: number }
export interface Recommendation { rank: number; scholarship_id: string; name: string; description: string; recommendation_score: number; semantic_similarity: number; major_match: number; eligible: boolean; academic_fit: number }
export interface RecommendationResponse { student_id: number; recommendations: Recommendation[] }
export interface Explanation { feature: string; impact: number; direction: "positive" | "negative" | "neutral"; description: string }
export interface Scholarship { scholarship_id: string; name: string; description: string; preferred_majors: string[]; academic_weight: number; extracurricular_weight: number; eligibility?: Record<string, unknown> }
export interface ApiError { message: string; errors?: Record<string, string> }
export interface User { id: number; name: string; email: string }
export interface AuthResponse { access_token: string; token_type: "bearer"; user: User }
