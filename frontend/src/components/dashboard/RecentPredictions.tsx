import React from 'react';
import { Link } from 'react-router-dom';
import { MdArrowForward } from 'react-icons/md';
import { Prediction } from '@/types';
import { Badge } from '@components/ui/Badge';
import { Card } from '@components/ui/Card';
import { Spinner } from '@components/ui/Spinner';

interface RecentPredictionsProps {
  predictions: Prediction[];
  isLoading: boolean;
}

export const RecentPredictions: React.FC<RecentPredictionsProps> = ({
  predictions,
  isLoading,
}) => {
  return (
    <Card className="col-span-full lg:col-span-2">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-base font-semibold text-white">Recent Predictions</h3>
        <Link
          to="/history"
          className="flex items-center gap-1 text-xs text-primary-400 hover:text-primary-300 transition-colors"
        >
          View All <MdArrowForward />
        </Link>
      </div>

      {isLoading ? (
        <div className="py-8 flex justify-center">
          <Spinner size="md" text="Loading predictions..." />
        </div>
      ) : predictions.length === 0 ? (
        <div className="py-10 text-center">
          <p className="text-sm text-white/30">No predictions yet.</p>
          <Link to="/predict" className="text-xs text-primary-400 hover:text-primary-300 mt-1 inline-block">
            Make your first prediction →
          </Link>
        </div>
      ) : (
        <div className="space-y-1">
          {predictions.slice(0, 5).map((p) => (
            <div
              key={p._id}
              className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/3 transition-colors group"
            >
              <Badge prediction={p.prediction} />
              <code className="flex-1 text-xs text-white/60 font-mono truncate group-hover:text-white/80 transition-colors">
                {p.smiles}
              </code>
              <span className="text-xs text-white/25 shrink-0">
                {new Date(p.createdAt).toLocaleDateString()}
              </span>
              {p.confidence !== null && (
                <span className="text-xs text-primary-400/70 shrink-0 hidden sm:block">
                  {Math.round(p.confidence * 100)}% conf
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};
