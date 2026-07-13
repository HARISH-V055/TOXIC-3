import React, { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { MdPerson, MdEmail, MdSave, MdLock, MdDeleteForever } from 'react-icons/md';
import { useAuthStore } from '@store/useAuthStore';
import { authService } from '@services/authService';
import { useNavigate } from 'react-router-dom';
import { Card } from '@components/ui/Card';
import { Input } from '@components/ui/Input';
import { Button } from '@components/ui/Button';
import { Alert } from '@components/ui/Alert';
import { AxiosError } from 'axios';

interface ProfileForm {
  name: string;
  email: string;
}

interface PasswordForm {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

const Profile: React.FC = () => {
  const { user, setUser, logout } = useAuthStore();
  const navigate = useNavigate();
  const [profileMsg, setProfileMsg] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);
  const [passwordMsg, setPasswordMsg] = useState<{ type: 'success' | 'error'; msg: string } | null>(null);
  const [predictionCount, setPredictionCount] = useState(0);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isSavingPassword, setIsSavingPassword] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const profileForm = useForm<ProfileForm>({
    defaultValues: { name: user?.name ?? '', email: user?.email ?? '' },
  });

  const passwordForm = useForm<PasswordForm>();
  const newPassword = passwordForm.watch('newPassword');

  useEffect(() => {
    authService.getProfile().then(({ user, predictionCount }) => {
      profileForm.reset({ name: user.name, email: user.email });
      setUser(user);
      setPredictionCount(predictionCount);
    });
  }, []);

  const onProfileSubmit = async (data: ProfileForm) => {
    setIsSavingProfile(true);
    setProfileMsg(null);
    try {
      const updated = await authService.updateProfile(data);
      setUser(updated);
      setProfileMsg({ type: 'success', msg: 'Profile updated successfully.' });
    } catch (err) {
      const error = err as AxiosError<{ message: string }>;
      setProfileMsg({ type: 'error', msg: error.response?.data?.message ?? 'Update failed.' });
    } finally {
      setIsSavingProfile(false);
    }
  };

  const onPasswordSubmit = async (data: PasswordForm) => {
    setIsSavingPassword(true);
    setPasswordMsg(null);
    try {
      await authService.changePassword(data);
      setPasswordMsg({ type: 'success', msg: 'Password changed. Please log in again.' });
      passwordForm.reset();
      setTimeout(() => { logout(); navigate('/login'); }, 2000);
    } catch (err) {
      const error = err as AxiosError<{ message: string }>;
      setPasswordMsg({ type: 'error', msg: error.response?.data?.message ?? 'Password change failed.' });
    } finally {
      setIsSavingPassword(false);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await authService.deleteAccount();
      logout();
      navigate('/');
    } catch {
      setProfileMsg({ type: 'error', msg: 'Failed to delete account.' });
    }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
          <MdPerson className="text-primary-400" />
          My Profile
        </h1>
        <p className="text-white/40 text-sm mt-1">Manage your account information</p>
      </div>

      {/* User Summary */}
      <Card className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center shrink-0 shadow-glow-sm text-xl font-bold text-white">
          {user?.name?.charAt(0)?.toUpperCase()}
        </div>
        <div>
          <p className="font-semibold text-white">{user?.name}</p>
          <p className="text-sm text-white/40">{user?.email}</p>
          <p className="text-xs text-white/30 mt-0.5">{predictionCount} predictions · {user?.role}</p>
        </div>
      </Card>

      {/* Profile Edit */}
      <Card>
        <h2 className="text-base font-semibold text-white mb-4">Personal Information</h2>
        {profileMsg && <Alert type={profileMsg.type} message={profileMsg.msg} className="mb-4" onClose={() => setProfileMsg(null)} />}
        <form onSubmit={profileForm.handleSubmit(onProfileSubmit)} className="space-y-4">
          <Input
            id="profile-name"
            label="Full Name"
            leftIcon={<MdPerson />}
            error={profileForm.formState.errors.name?.message}
            {...profileForm.register('name', { required: 'Name is required' })}
          />
          <Input
            id="profile-email"
            label="Email Address"
            type="email"
            leftIcon={<MdEmail />}
            error={profileForm.formState.errors.email?.message}
            {...profileForm.register('email', { required: 'Email is required', pattern: { value: /^\S+@\S+\.\S+$/, message: 'Invalid email' } })}
          />
          <Button id="profile-save-btn" type="submit" isLoading={isSavingProfile} leftIcon={<MdSave />}>
            Save Changes
          </Button>
        </form>
      </Card>

      {/* Change Password */}
      <Card>
        <h2 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
          <MdLock className="text-primary-400" /> Change Password
        </h2>
        {passwordMsg && <Alert type={passwordMsg.type} message={passwordMsg.msg} className="mb-4" onClose={() => setPasswordMsg(null)} />}
        <form onSubmit={passwordForm.handleSubmit(onPasswordSubmit)} className="space-y-4">
          <Input
            id="current-password"
            label="Current Password"
            type="password"
            leftIcon={<MdLock />}
            error={passwordForm.formState.errors.currentPassword?.message}
            {...passwordForm.register('currentPassword', { required: 'Current password is required' })}
          />
          <Input
            id="new-password"
            label="New Password"
            type="password"
            leftIcon={<MdLock />}
            error={passwordForm.formState.errors.newPassword?.message}
            hint="Min 8 chars with uppercase, lowercase, and number"
            {...passwordForm.register('newPassword', {
              required: 'New password is required',
              minLength: { value: 8, message: 'Must be at least 8 characters' },
              pattern: { value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, message: 'Must include uppercase, lowercase & number' },
            })}
          />
          <Input
            id="confirm-new-password"
            label="Confirm New Password"
            type="password"
            leftIcon={<MdLock />}
            error={passwordForm.formState.errors.confirmPassword?.message}
            {...passwordForm.register('confirmPassword', {
              required: 'Please confirm your password',
              validate: (v) => v === newPassword || 'Passwords do not match',
            })}
          />
          <Button id="change-password-btn" type="submit" variant="secondary" isLoading={isSavingPassword} leftIcon={<MdLock />}>
            Change Password
          </Button>
        </form>
      </Card>

      {/* Danger Zone */}
      <Card className="border-red-500/15">
        <h2 className="text-base font-semibold text-red-400 mb-2">Danger Zone</h2>
        <p className="text-xs text-white/40 mb-4">
          Deleting your account is permanent and will remove all your predictions and data.
        </p>
        {!showDeleteConfirm ? (
          <Button
            id="delete-account-btn"
            variant="danger"
            leftIcon={<MdDeleteForever />}
            onClick={() => setShowDeleteConfirm(true)}
          >
            Delete Account
          </Button>
        ) : (
          <div className="flex items-center gap-3">
            <Button variant="danger" leftIcon={<MdDeleteForever />} onClick={handleDeleteAccount}>
              Yes, Delete My Account
            </Button>
            <Button variant="ghost" onClick={() => setShowDeleteConfirm(false)}>
              Cancel
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
};

export default Profile;
