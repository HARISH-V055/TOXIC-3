import express, { Application } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import cookieParser from 'cookie-parser';
import { env } from './config/env';
import { logger } from './utils/logger';
import routes from './routes';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';
import { globalRateLimiter } from './middleware/rateLimiter';

const app: Application = express();

// ─── Security Middleware ───────────────────────────────────
app.use(
  helmet({
    crossOriginEmbedderPolicy: false,
    crossOriginResourcePolicy: { policy: 'cross-origin' },
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'blob:', 'http:', 'https:'],
      },
    },
  })
);

app.use(
  cors({
    origin: env.CORS_ORIGIN,
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

// ─── General Middleware ────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());
app.use(globalRateLimiter);

// ─── Logging ──────────────────────────────────────────────
const morganFormat = env.isDevelopment ? 'dev' : 'combined';
app.use(
  morgan(morganFormat, {
    stream: { write: (message) => logger.http(message.trim()) },
    skip: (_req, res) => env.isProduction && res.statusCode < 400,
  })
);

// ─── Health Check ─────────────────────────────────────────
app.get('/health', (_req, res) => {
  res.status(200).json({
    success: true,
    message: 'MolXAI API is healthy',
    timestamp: new Date().toISOString(),
    environment: env.NODE_ENV,
    version: '1.0.0',
  });
});

import path from 'path';

// ─── Static Outputs ────────────────────────────────────────
const outputsPath = path.join(__dirname, '../../EQ-KA-GCN/outputs');
app.use('/outputs', express.static(outputsPath));

// ─── API Routes ───────────────────────────────────────────
app.use('/api', routes);

// ─── 404 + Error Handling ─────────────────────────────────
app.use(notFoundHandler);
app.use(errorHandler);

export default app;
