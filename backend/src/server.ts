import { env } from './config/env';
import { connectDB, disconnectDB } from './config/db';
import { logger } from './utils/logger';
import app from './app';

const startServer = async (): Promise<void> => {
  await connectDB();

  const server = app.listen(env.PORT, () => {
    logger.info(`🚀 MolXAI Backend running in ${env.NODE_ENV} mode`);
    logger.info(`🌐 Server:    http://localhost:${env.PORT}`);
    logger.info(`❤️  Health:    http://localhost:${env.PORT}/health`);
    logger.info(`📡 API Base:  http://localhost:${env.PORT}/api`);
  });

  // ─── Graceful Shutdown ──────────────────────────────────
  const shutdown = async (signal: string): Promise<void> => {
    logger.info(`\n⚠️  Received ${signal}. Starting graceful shutdown...`);

    server.close(async () => {
      logger.info('HTTP server closed.');
      await disconnectDB();
      logger.info('✅ Graceful shutdown complete.');
      process.exit(0);
    });

    // Force exit if graceful shutdown takes too long
    setTimeout(() => {
      logger.error('Graceful shutdown timeout. Forcing exit.');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  process.on('unhandledRejection', (reason: unknown) => {
    logger.error('Unhandled Promise Rejection:', reason);
    shutdown('UNHANDLED_REJECTION');
  });

  process.on('uncaughtException', (error: Error) => {
    logger.error('Uncaught Exception:', error);
    shutdown('UNCAUGHT_EXCEPTION');
  });
};

startServer().catch((error) => {
  logger.error('Failed to start server:', error);
  process.exit(1);
});
