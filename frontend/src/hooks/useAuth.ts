import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@store/useAuthStore';
import { authService } from '@services/authService';
import { LoginCredentials, RegisterCredentials } from '@/types';

export const useAuth = () => {
  const navigate = useNavigate();
  const { user, accessToken, isAuthenticated, isLoading, setAuth, logout: storeLogout, setLoading } = useAuthStore();

  const login = useCallback(
    async (credentials: LoginCredentials) => {
      setLoading(true);
      try {
        const { user, accessToken } = await authService.login(credentials);
        setAuth(user, accessToken);
        navigate('/dashboard');
      } finally {
        setLoading(false);
      }
    },
    [setAuth, navigate, setLoading]
  );

  const register = useCallback(
    async (credentials: RegisterCredentials) => {
      setLoading(true);
      try {
        const { user, accessToken } = await authService.register(credentials);
        setAuth(user, accessToken);
        navigate('/dashboard');
      } finally {
        setLoading(false);
      }
    },
    [setAuth, navigate, setLoading]
  );

  const logout = useCallback(async () => {
    try {
      await authService.logout();
    } finally {
      storeLogout();
      navigate('/login');
    }
  }, [storeLogout, navigate]);

  return {
    user,
    accessToken,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
  };
};
