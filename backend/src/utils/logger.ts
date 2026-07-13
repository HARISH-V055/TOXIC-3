import winston from 'winston';
import { env } from '../config/env';

const { combine, timestamp, errors, json, colorize, printf, splat } = winston.format;

const devFormat = combine(
  colorize({ all: true }),
  timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  errors({ stack: true }),
  splat(),
  printf(({ timestamp, level, message, stack }) => {
    return stack
      ? `[${timestamp}] ${level}: ${message}\n${stack}`
      : `[${timestamp}] ${level}: ${message}`;
  })
);

const prodFormat = combine(
  timestamp(),
  errors({ stack: true }),
  splat(),
  json()
);

export const logger = winston.createLogger({
  level: env.LOG_LEVEL,
  format: env.isProduction ? prodFormat : devFormat,
  defaultMeta: { service: 'molxai-backend' },
  transports: [
    new winston.transports.Console(),
    ...(env.isProduction
      ? [
          new winston.transports.File({
            filename: 'logs/error.log',
            level: 'error',
            maxsize: 5242880,
            maxFiles: 5,
          }),
          new winston.transports.File({
            filename: 'logs/combined.log',
            maxsize: 5242880,
            maxFiles: 5,
          }),
        ]
      : []),
  ],
  exceptionHandlers: [new winston.transports.Console()],
  rejectionHandlers: [new winston.transports.Console()],
});
