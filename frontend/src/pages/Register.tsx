import React from 'react';
import { Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { MdEmail, MdLock, MdPerson } from 'react-icons/md';
import { useAuth } from '@hooks/useAuth';
import { RegisterCredentials } from '@/types';
import { Input } from '@components/ui/Input';
import { Button } from '@components/ui/Button';
import { Alert } from '@components/ui/Alert';
import { AxiosError } from 'axios';

interface RegisterForm extends RegisterCredentials {
  confirmPassword: string;
}

const Register: React.FC = () => {
  const { register: registerUser, isLoading } = useAuth();
  const [apiError, setApiError] = React.useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<RegisterForm>();

  const password = watch('password');

  const onSubmit = async ({ name, email, password }: RegisterForm) => {
    setApiError(null);
    try {
      await registerUser({ name, email, password });
    } catch (err) {
      const error = err as AxiosError<{ message: string }>;
      setApiError(error.response?.data?.message ?? 'Registration failed. Please try again.');
    }
  };

  return (
    <div className="glass-card p-8">
      <h2 className="text-xl font-bold text-white mb-1">Create your account</h2>
      <p className="text-sm text-white/40 mb-6">Join MolXAI — free forever for researchers</p>

      {apiError && (
        <Alert type="error" message={apiError} className="mb-5" onClose={() => setApiError(null)} />
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4" noValidate>
        <Input
          id="register-name"
          label="Full Name"
          type="text"
          placeholder="Dr. Jane Smith"
          leftIcon={<MdPerson />}
          error={errors.name?.message}
          {...register('name', {
            required: 'Name is required',
            minLength: { value: 2, message: 'Name must be at least 2 characters' },
          })}
        />

        <Input
          id="register-email"
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
          id="register-password"
          label="Password"
          type="password"
          placeholder="Min 8 chars, uppercase & number"
          leftIcon={<MdLock />}
          error={errors.password?.message}
          {...register('password', {
            required: 'Password is required',
            minLength: { value: 8, message: 'Password must be at least 8 characters' },
            pattern: {
              value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
              message: 'Must contain uppercase, lowercase, and a number',
            },
          })}
        />

        <Input
          id="register-confirm-password"
          label="Confirm Password"
          type="password"
          placeholder="Repeat your password"
          leftIcon={<MdLock />}
          error={errors.confirmPassword?.message}
          {...register('confirmPassword', {
            required: 'Please confirm your password',
            validate: (value) => value === password || 'Passwords do not match',
          })}
        />

        <Button
          id="register-submit-btn"
          type="submit"
          fullWidth
          isLoading={isLoading}
          className="mt-2"
        >
          Create Account
        </Button>
      </form>

      <p className="text-center text-sm text-white/40 mt-6">
        Already have an account?{' '}
        <Link to="/login" className="text-primary-400 hover:text-primary-300 font-medium transition-colors">
          Sign in
        </Link>
      </p>
    </div>
  );
};

export default Register;
