import React, { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { MdSearch, MdDelete, MdHistory, MdChevronLeft, MdChevronRight } from 'react-icons/md';
import { usePredictions } from '@hooks/usePredictions';
import { Prediction } from '@/types';
import { Badge } from '@components/ui/Badge';
import { Button } from '@components/ui/Button';
import { Spinner } from '@components/ui/Spinner';
import { Alert } from '@components/ui/Alert';

const LIMIT = 10;

const History: React.FC = () => {
  const { predictions, meta, isLoading, fetchPredictions, deletePrediction } = usePredictions();
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [filterResult, setFilterResult] = useState<string>('all');
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const loadPredictions = useCallback(() => {
    fetchPredictions({
      page,
      limit: LIMIT,
      search,
      sort: 'createdAt',
      order: 'desc',
    });
  }, [fetchPredictions, page, search]);

  useEffect(() => {
    loadPredictions();
  }, [loadPredictions]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    loadPredictions();
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this prediction?')) return;
    setDeletingId(id);
    try {
      await deletePrediction(id);
    } catch {
      setDeleteError('Failed to delete prediction.');
    } finally {
      setDeletingId(null);
    }
  };

  const filteredPredictions = filterResult === 'all'
    ? predictions
    : predictions.filter((p) => p.prediction === filterResult);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
            <MdHistory className="text-primary-400" />
            Prediction History
          </h1>
          <p className="text-white/40 text-sm mt-1">
            {meta?.total ?? 0} total predictions
          </p>
        </div>
      </div>

      {deleteError && (
        <Alert type="error" message={deleteError} onClose={() => setDeleteError(null)} />
      )}

      {/* Filters & Search */}
      <div className="flex items-center gap-3 flex-wrap">
        <form onSubmit={handleSearch} className="flex gap-2 flex-1 min-w-[200px] max-w-sm">
          <div className="relative flex-1">
            <MdSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30" />
            <input
              id="history-search"
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search SMILES..."
              className="input-field pl-9 text-sm"
            />
          </div>
          <Button type="submit" size="sm" variant="secondary">Search</Button>
        </form>

        <div className="flex gap-2">
          {['all', 'toxic', 'non-toxic', 'pending'].map((val) => (
            <button
              key={val}
              onClick={() => setFilterResult(val)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                filterResult === val
                  ? 'bg-primary-500/20 border border-primary-500/30 text-primary-300'
                  : 'bg-white/5 border border-white/10 text-white/50 hover:text-white hover:border-white/20'
              }`}
            >
              {val === 'all' ? 'All' : val.charAt(0).toUpperCase() + val.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="glass-card overflow-hidden p-0">
        {isLoading ? (
          <div className="py-16 flex justify-center">
            <Spinner size="lg" text="Loading predictions..." />
          </div>
        ) : filteredPredictions.length === 0 ? (
          <div className="py-16 text-center">
            <MdHistory className="text-4xl text-white/15 mx-auto mb-3" />
            <p className="text-sm text-white/30">No predictions found.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full" id="predictions-table">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs font-semibold text-white/40 px-6 py-4">SMILES</th>
                  <th className="text-left text-xs font-semibold text-white/40 px-4 py-4">Result</th>
                  <th className="text-right text-xs font-semibold text-white/40 px-4 py-4 hidden md:table-cell">Confidence</th>
                  <th className="text-right text-xs font-semibold text-white/40 px-4 py-4 hidden lg:table-cell">Date</th>
                  <th className="text-right text-xs font-semibold text-white/40 px-6 py-4">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((p: Prediction, i) => (
                  <motion.tr
                    key={p._id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.03 }}
                    className="border-b border-white/5 hover:bg-white/3 transition-colors group"
                  >
                    <td className="px-6 py-3.5">
                      <code className="text-xs font-mono text-primary-300/80 truncate block max-w-[240px] group-hover:text-primary-300 transition-colors">
                        {p.smiles}
                      </code>
                    </td>
                    <td className="px-4 py-3.5">
                      <Badge prediction={p.prediction} />
                    </td>
                    <td className="px-4 py-3.5 text-right text-xs text-white/50 hidden md:table-cell">
                      {p.confidence !== null ? `${Math.round(p.confidence * 100)}%` : '—'}
                    </td>
                    <td className="px-4 py-3.5 text-right text-xs text-white/35 hidden lg:table-cell">
                      {new Date(p.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-3.5 text-right">
                      <button
                        onClick={() => handleDelete(p._id)}
                        disabled={deletingId === p._id}
                        className="p-1.5 rounded-lg text-white/20 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200 disabled:opacity-50"
                        aria-label="Delete prediction"
                      >
                        {deletingId === p._id ? (
                          <span className="animate-spin">⏳</span>
                        ) : (
                          <MdDelete />
                        )}
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Pagination */}
      {meta && meta.pages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-xs text-white/30">
            Page {meta.page} of {meta.pages} · {meta.total} results
          </p>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              leftIcon={<MdChevronLeft />}
            >
              Prev
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setPage((p) => Math.min(meta.pages, p + 1))}
              disabled={page === meta.pages}
              rightIcon={<MdChevronRight />}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
