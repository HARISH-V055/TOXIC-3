import { Request } from 'express';

export interface JwtPayload {
  userId: string;
  email: string;
  role: UserRole;
  iat?: number;
  exp?: number;
}

export enum UserRole {
  USER = 'user',
  ADMIN = 'admin',
}

export interface AuthRequest extends Request {
  user?: JwtPayload;
}

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
  total?: number;
  page?: number;
  limit?: number;
  pages?: number;
}

export interface PaginationQuery {
  page?: string;
  limit?: string;
  search?: string;
  sort?: string;
  order?: 'asc' | 'desc';
}

export interface ImportantAtom {
  index: number;
  element: string;
  score: number;
}

export interface ImportantBond {
  source: number;
  target: number;
  score: number;
}

export interface MolecularGraph {
  atoms: { index: number; element: string; x: number; y: number }[];
  bonds: { source: number; target: number }[];
}

export interface AIPredictRequest {
  smiles: string;
}

export interface AIPredictResponse {
  smiles: string;
  prediction: string;
  probability: number;
  confidence: number;
  threshold: number;
  endpoint: string;
  inferenceTimeMs: number;
  importantAtoms: ImportantAtom[];
  importantBonds: ImportantBond[];
  molecularGraph: MolecularGraph;
  explanationImage: string;
}

export interface AIExplainRequest {
  smiles: string;
  predictionId?: string;
}

export interface AIExplainResponse {
  atomAttentions: { atomIndex: number; weight: number }[];
  bondAttentions: { bondIndex: number; weight: number; atoms: [number, number] }[];
  saliencyMap: number[][];
}
