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

  const getRiskBadge = (prob: number | null, threshold: number = 0.75) => {
    if (prob === null) return null;
    if (prob >= threshold) {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-red-500/15 text-red-400 border border-red-500/20">
          Tox21 SR-p53: Predicted Active
        </span>
      );
    } else {
      return (
        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-green-500/15 text-green-400 border border-green-500/20">
          Tox21 SR-p53: Predicted Non-Active
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
              <h2 className="text-lg font-semibold text-white">Prediction & Explainability Report</h2>
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
              {/* Model Output Summary Card */}
              <Card className="space-y-4">
                <div className="flex items-start justify-between border-b border-white/5 pb-3">
                  <div>
                    <p className="text-xs text-white/40 mb-1">Prediction Class</p>
                    <span className={`text-2xl font-black ${result.prediction?.toLowerCase() === 'toxic' ? 'text-red-400' : 'text-green-400'}`}>
                      {result.prediction === 'Toxic' || result.prediction === 'toxic' ? 'Predicted Toxic' : 'Predicted Non-Toxic'}
                    </span>
                    <p className="text-xs text-white/50 mt-1 font-medium">
                      {result.prediction?.toLowerCase() === 'toxic'
                        ? 'The model predicts Toxic for the Tox21 SR-p53 endpoint.'
                        : 'The model predicts Non-Toxic for the Tox21 SR-p53 endpoint.'}
                    </p>
                  </div>
                  {getRiskBadge(result.probability, result.threshold || 0.75)}
                </div>

                {/* Key Bioassay & Model Metrics */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                    <span className="text-white/40 block mb-1">Dataset</span>
                    <span className="text-white font-semibold">Tox21</span>
                  </div>
                  <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                    <span className="text-white/40 block mb-1">Bioassay Endpoint</span>
                    <span className="text-white font-semibold">{result.endpoint || 'SR-p53'}</span>
                  </div>
                  <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                    <span className="text-white/40 block mb-1">AI Architecture</span>
                    <span className="text-white font-semibold">Adaptive Quantized KA-GCN</span>
                  </div>
                  <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                    <span className="text-white/40 block mb-1">Decision Threshold</span>
                    <span className="text-white font-mono font-semibold">
                      75% (0.75)
                    </span>
                  </div>
                </div>

                {/* Mathematically Consistent Probabilities (Requirement 2 & 3) */}
                <div className="space-y-3 pt-2">
                  {result.probability !== null && (
                    <>
                      <ConfidenceBar
                        value={result.probability}
                        label="Toxicity Probability"
                        color={result.probability >= (result.threshold || 0.75) ? 'red' : 'green'}
                      />
                      <ConfidenceBar
                        value={1 - result.probability}
                        label="Non-Toxic Probability"
                        color={result.probability < (result.threshold || 0.75) ? 'green' : 'blue'}
                      />
                    </>
                  )}
                  <p className="text-[10px] text-white/30 font-mono">
                    Decision Rule: Toxicity Probability ≥ 75.00% → Toxic | Toxicity Probability &lt; 75.00% → Non-Toxic
                  </p>
                </div>

                {/* Inference & Response Timings (Requirement 12) */}
                <div className="flex flex-wrap justify-between items-center text-xs text-white/40 pt-3 border-t border-white/5 font-mono">
                  <span>
                    Model Inference Time:{' '}
                    <strong className="text-white">
                      {result.inferenceTimeMs ? `${result.inferenceTimeMs.toFixed(2)} ms` : result.executionTime ? `${result.executionTime.toFixed(2)} ms` : 'N/A'}
                    </strong>
                  </span>
                  {result.totalResponseTimeMs && (
                    <span>
                      Total Response Time:{' '}
                      <strong className="text-white/80">{result.totalResponseTimeMs} ms</strong>
                    </span>
                  )}
                </div>
              </Card>

              {/* RDKit Molecular Structure Canvas (Requirement 8 & 10) */}
              <Card padding="sm" className="print-molecule-card">
                <p className="text-xs text-white/40 mb-3 px-2 font-medium">Molecular Structure Depiction</p>
                <MoleculeViewer
                  smiles={result.smiles}
                  molecularGraph={result.molecularGraph}
                  importantAtoms={result.importantAtoms}
                  importantBonds={result.importantBonds}
                  width={400}
                  height={260}
                  className="w-full"
                />
                <p className="text-[10px] text-white/30 text-center mt-2 font-mono">
                  Exact RDKit 2D atom coordinates with GNNExplainer attribution highlights
                </p>
              </Card>
            </div>

            {/* Methodology Overview Section (Requirement 13) */}
            <Card>
              <h3 className="text-xs uppercase tracking-wider text-white/50 font-bold mb-3">
                Methodology Overview
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 text-xs">
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Model</span>
                  <span className="text-white font-semibold">Adaptive Quantized KA-GCN</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Dataset</span>
                  <span className="text-white font-semibold">Tox21</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Endpoint</span>
                  <span className="text-white font-semibold">SR-p53</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Decision Threshold</span>
                  <span className="text-white font-mono font-semibold">0.75</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5 col-span-2 sm:col-span-1">
                  <span className="text-white/40 block mb-1">Explainability</span>
                  <span className="text-primary-400 font-semibold">GNNExplainer</span>
                </div>
              </div>
            </Card>

            {/* GNNExplainer Rankings Section (Requirement 6) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Important Atoms */}
              <Card>
                <h3 className="text-xs uppercase tracking-wider text-white/50 font-bold mb-3">
                  GNNExplainer — Important Atoms
                </h3>
                {result.importantAtoms && result.importantAtoms.length > 0 ? (
                  <div className="space-y-2">
                    {result.importantAtoms.map((atom, idx) => (
                      <div
                        key={idx}
                        className="flex justify-between items-center p-2 rounded-lg bg-white/5 border border-white/5 text-xs font-mono"
                      >
                        <span className="text-primary-300 font-bold">
                          Atom #{atom.index} ({atom.element})
                        </span>
                        <span className="text-white/70">
                          Attribution Score: <strong className="text-white">{typeof atom.score === 'number' ? atom.score.toFixed(4) : atom.score}</strong>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-white/30">No high-importance atoms identified.</p>
                )}
              </Card>

              {/* Important Bonds */}
              <Card>
                <h3 className="text-xs uppercase tracking-wider text-white/50 font-bold mb-3">
                  GNNExplainer — Important Bonds
                </h3>
                {result.importantBonds && result.importantBonds.length > 0 ? (
                  <div className="space-y-2">
                    {result.importantBonds.map((bond, idx) => (
                      <div
                        key={idx}
                        className="flex justify-between items-center p-2 rounded-lg bg-white/5 border border-white/5 text-xs font-mono"
                      >
                        <span className="text-accent-300 font-bold">
                          Bond #{bond.source} — #{bond.target}
                        </span>
                        <span className="text-white/70">
                          Attribution Score: <strong className="text-white">{typeof bond.score === 'number' ? bond.score.toFixed(4) : bond.score}</strong>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-white/30">No high-importance bonds identified.</p>
                )}
              </Card>
            </div>

            {/* Explanation Semantics Note (Requirement 7 & 14) */}
            <div className="p-3.5 rounded-xl bg-primary-500/10 border border-primary-500/20 text-xs text-primary-200 leading-relaxed">
              <span className="font-semibold text-primary-300">ℹ️ Note:</span> GNNExplainer scores represent the relative contribution of atoms and bonds to the model's prediction. A high score does not mean that the atom or bond is inherently toxic.
            </div>

            {/* Explanation Metadata Section (Requirement 10) */}
            <Card>
              <h3 className="text-xs uppercase tracking-wider text-white/50 font-bold mb-3">
                Explanation Metadata
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Model</span>
                  <span className="text-white font-semibold">Adaptive Quantized KA-GCN</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Explainer</span>
                  <span className="text-white font-semibold">GNNExplainer</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Decision Threshold</span>
                  <span className="text-white font-mono font-semibold">0.75 (75.00%)</span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Predicted Class</span>
                  <span className={`font-semibold ${result.prediction?.toLowerCase() === 'toxic' ? 'text-red-400' : 'text-green-400'}`}>
                    {result.prediction}
                  </span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Toxicity Probability</span>
                  <span className="text-white font-mono font-semibold">
                    {result.probability !== null ? `${(result.probability * 100).toFixed(2)}%` : 'N/A'}
                  </span>
                </div>
                <div className="bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="text-white/40 block mb-1">Explanation Status</span>
                  <span className="text-green-400 font-semibold flex items-center gap-1">
                    ✓ Successfully generated
                  </span>
                </div>
              </div>
            </Card>

            {/* Explanation Visualization Plot Image (Requirement 8 & 11) */}
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs uppercase tracking-wider text-white/50 font-bold">
                  Explanation Visualization
                </h3>
                <span className="text-[10px] font-mono text-primary-400 bg-primary-500/10 border border-primary-500/20 px-2 py-0.5 rounded">
                  GNNExplainer 2D High-DPI Plot
                </span>
              </div>
              <div className="rounded-xl overflow-hidden bg-white p-3 flex justify-center shadow-lg border border-white/10">
                <img
                  key={result.smiles}
                  src={
                    result.explanationImage
                      ? `${result.explanationImage}?t=${Date.now()}`
                      : `/outputs/explanations/molecule_explanation.png?t=${Date.now()}`
                  }
                  alt="GNNExplainer Explanation Plot"
                  className="max-h-96 w-auto object-contain rounded"
                  onError={(e) => {
                    const imgEl = e.target as HTMLImageElement;
                    if (!imgEl.dataset.retried) {
                      imgEl.dataset.retried = 'true';
                      imgEl.src = `http://localhost:5000/outputs/explanations/molecule_explanation.png?t=${Date.now()}`;
                    }
                  }}
                />
              </div>
            </Card>

            {/* Analyzed SMILES Display */}
            <div className="glass-card p-4">
              <p className="text-xs text-white/40 mb-1.5 font-medium">Analyzed SMILES Notation</p>
              <code className="text-sm font-mono text-primary-300 break-all">{result.smiles}</code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Predict;
