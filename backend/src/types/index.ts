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
  name?: string;
  score: number;
  rank?: number;
  influenceType?: string;
  role?: string;
  description?: string;
}

export interface ImportantBond {
  source: number;
  target: number;
  score: number;
  rank?: number;
  bondName?: string;
  influenceType?: string;
  role?: string;
  description?: string;
}

export interface MolecularGraph {
  atoms: { index: number; element: string; x: number; y: number }[];
  bonds: { source: number; target: number }[];
}

export interface AIPredictRequest {
  smiles: string;
}

export interface EndpointPrediction {
  endpoint: string;
  name: string;
  category: string;
  prediction: string;
  probability: number;
  confidence: number;
  threshold: number;
}

export interface AIPredictResponse {
  smiles: string;
  prediction: string;
  probability: number;
  confidence: number;
  threshold: number;
  endpoint: string;
  inferenceTimeMs: number;
  endpoints?: EndpointPrediction[];
  importantAtoms: ImportantAtom[];
  importantBonds: ImportantBond[];
  explanationSummary?: string;
  molecularGraph: MolecularGraph;
  explanationImage: string;
}

export interface AIExplainRequest {
  smiles: string;
  predictionId?: string;
  targetEndpoint?: string;
}

export interface AIExplainResponse {
  atomAttentions: { atomIndex: number; weight: number }[];
  bondAttentions: { bondIndex: number; weight: number; atoms: [number, number] }[];
  saliencyMap: number[][];
}
