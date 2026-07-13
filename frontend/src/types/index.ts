// User types
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'user' | 'admin';
  createdAt?: string;
  updatedAt?: string;
}

// Auth types
export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterCredentials {
  name: string;
  email: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  accessToken: string;
}

// Prediction types
export interface ImportantBond {
  atomA: number;
  atomB: number;
  weight: number;
}

export type PredictionResult = 'toxic' | 'non-toxic' | 'pending' | 'error';

export interface Prediction {
  _id: string;
  user: string;
  smiles: string;
  prediction: PredictionResult;
  probability: number | null;
  confidence: number | null;
  importantAtoms: number[];
  importantBonds: ImportantBond[];
  executionTime: number | null;
  createdAt: string;
  updatedAt: string;
}

export interface PredictionStats {
  total: number;
  toxic: number;
  nonToxic: number;
  pending: number;
  avgConfidence: number;
  avgExecutionTime: number;
}

// API types
export interface ApiResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
  errors?: ValidationError[];
  meta?: ResponseMeta;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface ResponseMeta {
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
  search?: string;
  sort?: string;
  order?: 'asc' | 'desc';
}

// Model types
export interface ModelInfo {
  version: string;
  name: string;
  description: string;
  accuracy?: number;
  status: 'active' | 'training' | 'offline';
}

export interface ModelStatus {
  service: { status: 'online' | 'offline'; version?: string };
  model: ModelInfo;
}

// UI types
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}
