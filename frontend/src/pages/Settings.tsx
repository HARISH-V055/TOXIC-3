import React, { useState } from 'react';
import { MdSettings, MdPalette, MdNotifications, MdSecurity } from 'react-icons/md';
import { Card } from '@components/ui/Card';
import { Button } from '@components/ui/Button';
import { Alert } from '@components/ui/Alert';

const Settings: React.FC = () => {
  const [saved, setSaved] = useState(false);
  const [notifications, setNotifications] = useState({
    emailOnPrediction: true,
    weeklyReport: false,
    systemAlerts: true,
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
          <MdSettings className="text-primary-400" />
          Settings
        </h1>
        <p className="text-white/40 text-sm mt-1">Manage your preferences and application settings</p>
      </div>

      {saved && <Alert type="success" message="Settings saved successfully." />}

      {/* Theme Settings */}
      <Card>
        <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-4">
          <MdPalette className="text-primary-400" /> Appearance
        </h2>
        <div className="space-y-4">
          <div>
            <label className="form-label">Theme</label>
            <div className="grid grid-cols-3 gap-3">
              {['Dark (Default)', 'Dark Blue', 'Midnight'].map((theme, i) => (
                <button
                  key={theme}
                  className={`p-3 rounded-xl border text-xs font-medium transition-all duration-200 ${
                    i === 0
                      ? 'border-primary-500/50 bg-primary-500/10 text-primary-400'
                      : 'border-white/10 bg-white/3 text-white/40 hover:border-white/20 hover:text-white/70'
                  }`}
                >
                  {theme}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="form-label">Accent Color</label>
            <div className="flex gap-3">
              {[
                { name: 'Cyan', from: 'from-primary-500', to: 'to-accent-500' },
                { name: 'Violet', from: 'from-violet-500', to: 'to-purple-500' },
                { name: 'Green', from: 'from-green-500', to: 'to-emerald-500' },
                { name: 'Orange', from: 'from-orange-500', to: 'to-amber-500' },
              ].map(({ name, from, to }, i) => (
                <button
                  key={name}
                  aria-label={`Accent ${name}`}
                  className={`w-8 h-8 rounded-lg bg-gradient-to-br ${from} ${to} ${i === 0 ? 'ring-2 ring-white/30 ring-offset-2 ring-offset-surface-800' : ''} transition-all hover:scale-110`}
                />
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Notifications */}
      <Card>
        <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-4">
          <MdNotifications className="text-primary-400" /> Notifications
        </h2>
        <div className="space-y-4">
          {[
            { key: 'emailOnPrediction', label: 'Email on Prediction Complete', desc: 'Receive an email when a prediction finishes' },
            { key: 'weeklyReport', label: 'Weekly Summary Report', desc: 'Get a weekly digest of your prediction activity' },
            { key: 'systemAlerts', label: 'System Alerts', desc: 'Receive alerts for model updates and outages' },
          ].map(({ key, label, desc }) => (
            <div key={key} className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-white">{label}</p>
                <p className="text-xs text-white/40 mt-0.5">{desc}</p>
              </div>
              <button
                onClick={() => setNotifications((n) => ({ ...n, [key]: !n[key as keyof typeof n] }))}
                className={`relative w-10 h-5 rounded-full transition-all duration-300 shrink-0 ${
                  notifications[key as keyof typeof notifications]
                    ? 'bg-primary-500'
                    : 'bg-white/10'
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-300 ${
                    notifications[key as keyof typeof notifications] ? 'translate-x-5' : ''
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </Card>

      {/* Security */}
      <Card>
        <h2 className="text-base font-semibold text-white flex items-center gap-2 mb-4">
          <MdSecurity className="text-primary-400" /> Security
        </h2>
        <div className="space-y-3 text-sm text-white/60">
          <p>• Sessions are managed with HttpOnly refresh tokens</p>
          <p>• Passwords are hashed with bcrypt (12 rounds)</p>
          <p>• Access tokens expire every 15 minutes</p>
          <p>• All API endpoints protected by rate limiting</p>
        </div>
      </Card>

      <Button id="settings-save-btn" onClick={handleSave}>Save All Settings</Button>
    </div>
  );
};

export default Settings;
