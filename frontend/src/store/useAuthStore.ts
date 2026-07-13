import { create } from 'zustand';
import { persist, devtools } from 'zustand/middleware';
import { User } from '@/types';

interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  predictionCount: number;

  // Actions
  setAuth: (user: User, accessToken: string) => void;
  setAccessToken: (token: string) => void;
  setUser: (user: User) => void;
  setPredictionCount: (count: number) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        accessToken: null,
        isAuthenticated: false,
        isLoading: false,
        predictionCount: 0,

        setAuth: (user, accessToken) =>
          set({ user, accessToken, isAuthenticated: true }, false, 'setAuth'),

        setAccessToken: (accessToken) =>
          set({ accessToken }, false, 'setAccessToken'),

        setUser: (user) =>
          set({ user }, false, 'setUser'),

        setPredictionCount: (predictionCount) =>
          set({ predictionCount }, false, 'setPredictionCount'),

        logout: () =>
          set(
            {
              user: null,
              accessToken: null,
              isAuthenticated: false,
              predictionCount: 0,
            },
            false,
            'logout'
          ),

        setLoading: (isLoading) =>
          set({ isLoading }, false, 'setLoading'),
      }),
      {
        name: 'molxai-auth',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
          // Note: accessToken is intentionally NOT persisted for security
          // It is refreshed on page load via the refresh token (HttpOnly cookie)
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);
