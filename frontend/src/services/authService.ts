import api from './api';
import {
  ApiResponse,
  AuthResponse,
  LoginCredentials,
  RegisterCredentials,
  User,
} from '@/types';

export const authService = {
  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    const { data } = await api.post<ApiResponse<AuthResponse>>('/auth/register', credentials);
    return data.data!;
  },

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const { data } = await api.post<ApiResponse<AuthResponse>>('/auth/login', credentials);
    return data.data!;
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },

  async refreshToken(): Promise<{ accessToken: string }> {
    const { data } = await api.post<ApiResponse<{ accessToken: string }>>('/auth/refresh');
    return data.data!;
  },

  async getProfile(): Promise<{ user: User; predictionCount: number }> {
    const { data } = await api.get<ApiResponse<{ user: User; predictionCount: number }>>('/user/profile');
    return data.data!;
  },

  async updateProfile(updates: Partial<{ name: string; email: string }>): Promise<User> {
    const { data } = await api.put<ApiResponse<{ user: User }>>('/user/profile', updates);
    return data.data!.user;
  },

  async changePassword(payload: {
    currentPassword: string;
    newPassword: string;
    confirmPassword: string;
  }): Promise<void> {
    await api.put('/user/password', payload);
  },

  async deleteAccount(): Promise<void> {
    await api.delete('/user');
  },
};
