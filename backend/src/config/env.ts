import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../../.env') });

const getEnvVar = (key: string, fallback?: string): string => {
  const value = process.env[key] ?? fallback;
  if (value === undefined) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
  return value;
};

export const env = {
  NODE_ENV: getEnvVar('NODE_ENV', 'development'),
  PORT: parseInt(getEnvVar('PORT', '5000'), 10),

  MONGO_URI: getEnvVar('MONGO_URI', 'mongodb://localhost:27017/molxai'),

  JWT_ACCESS_SECRET: getEnvVar('JWT_ACCESS_SECRET', 'dev_access_secret_change_in_production'),
  JWT_REFRESH_SECRET: getEnvVar('JWT_REFRESH_SECRET', 'dev_refresh_secret_change_in_production'),
  JWT_ACCESS_EXPIRES_IN: getEnvVar('JWT_ACCESS_EXPIRES_IN', '15m'),
  JWT_REFRESH_EXPIRES_IN: getEnvVar('JWT_REFRESH_EXPIRES_IN', '7d'),

  BCRYPT_SALT_ROUNDS: parseInt(getEnvVar('BCRYPT_SALT_ROUNDS', '12'), 10),

  CORS_ORIGIN: getEnvVar('CORS_ORIGIN', 'http://localhost:5173'),

  RATE_LIMIT_WINDOW_MS: parseInt(getEnvVar('RATE_LIMIT_WINDOW_MS', '900000'), 10),
  RATE_LIMIT_MAX_REQUESTS: parseInt(getEnvVar('RATE_LIMIT_MAX_REQUESTS', '100'), 10),

  AI_SERVICE_URL: getEnvVar('AI_SERVICE_URL', 'http://localhost:8000'),

  LOG_LEVEL: getEnvVar('LOG_LEVEL', 'info'),

  get isDevelopment() {
    return this.NODE_ENV === 'development';
  },
  get isProduction() {
    return this.NODE_ENV === 'production';
  },
} as const;
