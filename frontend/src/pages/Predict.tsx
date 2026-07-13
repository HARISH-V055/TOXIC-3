import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { MdScience, MdClear } from 'react-icons/md';
import { TbBolt, TbDownload } from 'react-icons/tb';
import { usePredictions } from '@hooks/usePredictions';
import { Prediction } from '@/types';
import { Button } from '@components/ui/Button';
import { Alert } from '@components/ui/Alert';
import { Badge } from '@components/ui/Badge';
import { Card } from '@components/ui/Card';
import { MoleculeViewer } from '@components/molecule/MoleculeViewer';
import { ConfidenceBar } from '@components/prediction/PredictionCard';

interface PredictForm {
  smiles: string;
}

const SAMPLE_SMILES = [
  { label: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { label: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { label: 'Ethanol', smiles: 'CCO' },
  { label: 'Benzene', smiles: 'c1ccccc1' },
];

const Predict: React.FC = () => {
  const { predict, isPredicting, error } = usePredictions();
  const [result, setResult] = useState<Prediction | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    reset,
    formState: { errors },
  } = useForm<PredictForm>();

  const smilesValue = watch('smiles', '');

  const onSubmit = async ({ smiles }: PredictForm) => {
    setApiError(null);
    setResult(null);
    try {
      const prediction = await predict(smiles.trim());
      setResult(prediction);
    } catch {
      setApiError(error ?? 'Prediction failed. Please try again.');
    }
  };

  const handleClear = () => {
    reset();
    setResult(null);
    setApiError(null);
  };

  const getRiskBadge = (prob: number | null) => {
    if (prob === null) return null;
    if (prob < 0.35) {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-green-500/15 text-green-400 border border-green-500/20">
          Low Toxicity Risk
        </span>
      );
    } else if (prob < 0.70) {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-yellow-500/15 text-yellow-400 border border-yellow-500/20">
          Medium Toxicity Risk
        </span>
      );
    } else {
      return (
        <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-red-500/15 text-red-400 border border-red-500/20">
          High Toxicity Risk
        </span>
      );
    }
  };

  const handleDownloadPDF = () => {
    window.print();
  };

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex justify-between items-start gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-white flex items-center gap-2">
            <MdScience className="text-primary-400" />
            Toxicity Prediction
          </h1>
          <p className="text-white/40 text-sm mt-1">
            Predict toxicity characteristics using the **EQ-KA-GCN** Kolmogorov-Arnold GNN.
          </p>
        </div>
        <div className="text-right">
          <span className="text-[10px] text-white/30 uppercase block font-semibold">Model Version</span>
          <span className="px-2.5 py-1 mt-1 inline-block rounded-lg bg-white/5 border border-white/10 text-xs text-white/70 font-mono">
            0.1.0-quantized
          </span>
        </div>
      </div>

      {/* Quick Sample Buttons */}
      <div className="flex flex-wrap gap-2">
        {SAMPLE_SMILES.map(({ label, smiles }) => (
          <button
            key={label}
            onClick={() => setValue('smiles', smiles)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 border border-white/10 text-white/60 hover:text-white hover:border-primary-500/30 hover:bg-primary-500/5 transition-all duration-200"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Input Form */}
      <Card className="no-print">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label htmlFor="smiles-input" className="form-label">
              SMILES String
            </label>
            <div className="relative">
              <textarea
                id="smiles-input"
                rows={4}
                placeholder="Enter SMILES notation, e.g. CC(=O)Oc1ccccc1C(=O)O"
                className={`
                  input-field font-mono text-sm resize-none
                  ${errors.smiles ? 'border-red-500/50 focus:ring-red-500/50' : ''}
                `}
                {...register('smiles', {
                  required: 'SMILES string is required',
                  minLength: { value: 1, message: 'SMILES string cannot be empty' },
                  maxLength: { value: 10000, message: 'SMILES string is too long' },
                })}
              />
              {smilesValue && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="absolute top-3 right-3 p-1 rounded-md text-white/30 hover:text-white hover:bg-white/10 transition-colors"
                  aria-label="Clear input"
                >
                  <MdClear />
                </button>
              )}
            </div>
            {errors.smiles && (
              <p className="mt-1.5 text-xs text-red-400 flex items-center gap-1">
                <span>⚠</span> {errors.smiles.message}
              </p>
            )}
            {smilesValue && !errors.smiles && (
              <p className="mt-1.5 text-xs text-white/30">
                {smilesValue.length} characters · {smilesValue.replace(/[^A-Z]/g, '').length} heavy atoms
              </p>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Button
              id="predict-submit-btn"
              type="submit"
              isLoading={isPredicting}
              leftIcon={<TbBolt />}
              disabled={!smilesValue}
            >
              {isPredicting ? 'Analyzing...' : 'Predict Toxicity'}
            </Button>
            {(result || smilesValue) && (
              <Button
                type="button"
                variant="ghost"
                onClick={handleClear}
                leftIcon={<MdClear />}
              >
                Clear
              </Button>
            )}
          </div>
        </form>
      </Card>

      {/* Error Display */}
      {apiError && (
        <Alert type="error" message={apiError} onClose={() => setApiError(null)} />
      )}

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.4 }}
            className="space-y-5 print-container"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Prediction Result</h2>
              <Button
                onClick={handleDownloadPDF}
                variant="secondary"
                size="sm"
                leftIcon={<TbDownload />}
                className="no-print"
              >
                Export Report / PDF
              </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Result Card */}
              <Card>
                <div className="flex items-start justify-between mb-5">
                  <div>
                    <p className="text-xs text-white/40 mb-1">Result</p>
                    <div className="flex items-center gap-3">
                      <Badge prediction={result.prediction} />
                      <span className={`text-2xl font-black ${result.prediction === 'toxic' ? 'text-red-400' : 'text-green-400'}`}>
                        {result.prediction === 'toxic' ? '⚠ Toxic' : '✓ Safe'}
                      </span>
                    </div>
                  </div>
                  {result.executionTime && (
                    <span className="text-xs text-white/30 bg-white/5 px-2 py-1 rounded-lg">
                      {result.executionTime.toFixed(0)}ms
                    </span>
                  )}
                </div>

                <div className="space-y-4">
                  {/* Risk Badge */}
                  <div className="flex justify-between items-center text-xs border-b border-white/5 pb-2">
                    <span className="text-white/40">Risk Metric</span>
                    {getRiskBadge(result.probability)}
                  </div>

                  {result.probability !== null && (
                    <ConfidenceBar
                      value={result.probability}
                      label="Toxicity Probability"
                      color={result.prediction === 'toxic' ? 'red' : 'green'}
                    />
                  )}
                  {result.confidence !== null && (
                    <ConfidenceBar
                      value={result.confidence}
                      label="Model Confidence"
                      color="primary"
                    />
                  )}
                </div>

                {/* Atom highlights */}
                {result.importantAtoms.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <p className="text-xs text-white/40 mb-2">Important Atoms (GNN Attention)</p>
                    <div className="flex flex-wrap gap-1.5">
                      {result.importantAtoms.map((atomIdx) => (
                        <span
                          key={atomIdx}
                          className="px-2 py-0.5 rounded-md bg-primary-500/15 border border-primary-500/20 text-primary-400 text-xs font-mono"
                        >
                          Atom #{atomIdx}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Important bonds */}
                {result.importantBonds.length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs text-white/40 mb-2">Important Bonds</p>
                    <div className="flex flex-wrap gap-1.5">
                      {result.importantBonds.map((bond, i) => (
                        <span
                          key={i}
                          className="px-2 py-0.5 rounded-md bg-accent-500/10 border border-accent-500/20 text-accent-400 text-xs font-mono"
                        >
                          #{bond.atomA}–#{bond.atomB} ({Math.round(bond.weight * 100)}%)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </Card>

              {/* Molecule Viewer */}
              <Card padding="sm" className="print-molecule-card">
                <p className="text-xs text-white/40 mb-3 px-2">Molecular Structure</p>
                <MoleculeViewer
                  smiles={result.smiles}
                  importantAtoms={result.importantAtoms}
                  importantBonds={result.importantBonds}
                  width={400}
                  height={280}
                  className="w-full"
                />
                <p className="text-[10px] text-white/20 text-center mt-2">
                  Highlighted atoms/bonds indicate high GNN attention weights
                </p>
              </Card>
            </div>

            {/* SMILES display */}
            <div className="glass-card p-4">
              <p className="text-xs text-white/40 mb-1.5">Analyzed SMILES</p>
              <code className="text-sm font-mono text-primary-300 break-all">{result.smiles}</code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Predict;
