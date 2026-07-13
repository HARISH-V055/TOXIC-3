import React from 'react';
import { Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { MdEmail, MdLock } from 'react-icons/md';
import { useAuth } from '@hooks/useAuth';
import { LoginCredentials } from '@/types';
import { Input } from '@components/ui/Input';
import { Button } from '@components/ui/Button';
import { Alert } from '@components/ui/Alert';
import { AxiosError } from 'axios';

const Login: React.FC = () => {
  const { login, isLoading } = useAuth();
  const [apiError, setApiError] = React.useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginCredentials>();

  const onSubmit = async (data: LoginCredentials) => {
    setApiError(null);
    try {
      await login(data);
    } catch (err) {
      const error = err as AxiosError<{ message: string }>;
      setApiError(error.response?.data?.message ?? 'Login failed. Please try again.');
    }
  };

  return (
    <div className="glass-card p-8">
      <h2 className="text-xl font-bold text-white mb-1">Welcome back</h2>
      <p className="text-sm text-white/40 mb-6">Sign in to your MolXAI account</p>

      {apiError && (
        <Alert type="error" message={apiError} className="mb-5" onClose={() => setApiError(null)} />
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4" noValidate>
        <Input
          id="login-email"
          label="Email Address"
          type="email"
          placeholder="you@example.com"
          leftIcon={<MdEmail />}
          error={errors.email?.message}
          {...register('email', {
            required: 'Email is required',
            pattern: { value: /^\S+@\S+\.\S+$/, message: 'Invalid email address' },
          })}
        />

        <Input
          id="login-password"
          label="Password"
          type="password"
          placeholder="••••••••"
          leftIcon={<MdLock />}
          error={errors.password?.message}
          {...register('password', {
            required: 'Password is required',
          })}
        />

        <Button
          id="login-submit-btn"
          type="submit"
          fullWidth
          isLoading={isLoading}
          className="mt-2"
        >
          Sign In
        </Button>
      </form>

      <p className="text-center text-sm text-white/40 mt-6">
        Don't have an account?{' '}
        <Link to="/register" className="text-primary-400 hover:text-primary-300 font-medium transition-colors">
          Create one free
        </Link>
      </p>
    </div>
  );
};

export default Login;
