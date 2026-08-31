import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { MdScience, MdTrendingUp, MdLayers, MdBook, MdMenuBook, MdCloudUpload } from 'react-icons/md';
import { TbArrowRight } from 'react-icons/tb';
import { useAuthStore } from '@store/useAuthStore';
import { usePredictions } from '@hooks/usePredictions';
import { StatsCard } from '@components/dashboard/StatsCard';
import { RecentPredictions } from '@components/dashboard/RecentPredictions';
import { ModelStatusCard } from '@components/dashboard/ModelStatus';
import { Card } from '@components/ui/Card';

const Dashboard: React.FC = () => {
  const user = useAuthStore((s) => s.user);
  const { predictions, stats, modelStatus, isLoading, fetchPredictions, fetchStats, fetchModelStatus } = usePredictions();

  useEffect(() => {
    fetchPredictions({ limit: 10 });
    fetchStats();
    fetchModelStatus();
  }, [fetchPredictions, fetchStats, fetchModelStatus]);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      alert(`File "${e.target.files[0].name}" uploaded. Extracting chemical compounds...`);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-white">
          {getGreeting()},{' '}
          <span className="gradient-text">{user?.name?.split(' ')[0] ?? 'Researcher'}</span> 👋
        </h1>
        <p className="text-white/40 text-sm mt-1">
          Welcome to the **EQ-KA-GCN** Model Validation & Prediction Platform.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatsCard
          title="Prediction Accuracy"
          value="88.9%"
          subtitle="Tox21 validation"
          icon={<MdTrendingUp />}
          iconBg="from-green-500 to-emerald-600"
          delay={0}
        />
        <StatsCard
          title="Total Predictions"
          value={stats?.total ?? 0}
          subtitle="All time logs"
          icon={<MdScience />}
          iconBg="from-primary-500 to-accent-500"
          delay={0.05}
        />
        <StatsCard
          title="Research Model"
          value="EQ-KA-GCN"
          subtitle="Kolmogorov-Arnold GNN"
          icon={<MdLayers />}
          iconBg="from-violet-500 to-purple-600"
          delay={0.1}
        />
      </div>

      {/* Model Metadata row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-white/2 border border-white/5 p-4 rounded-2xl text-xs text-white/50">
        <div>
          <span className="text-[10px] text-white/30 uppercase block font-semibold">Current Dataset</span>
          <span className="text-white font-medium mt-0.5 block">Tox21 / ClinTox</span>
        </div>
        <div>
          <span className="text-[10px] text-white/30 uppercase block font-semibold">Latest Training</span>
          <span className="text-white font-medium mt-0.5 block">24 hours ago</span>
        </div>
        <div>
          <span className="text-[10px] text-white/30 uppercase block font-semibold">Model Version</span>
          <span className="text-white font-medium mt-0.5 block">0.1.0-quantized</span>
        </div>
        <div>
          <span className="text-[10px] text-white/30 uppercase block font-semibold">Compression Ratio</span>
          <span className="text-white font-medium mt-0.5 block">85.4% (INT8)</span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <RecentPredictions predictions={predictions} isLoading={isLoading} />

        <div className="space-y-4">
          <ModelStatusCard status={modelStatus} />

          {/* Quick Actions Card */}
          <Card>
            <h3 className="text-sm font-semibold text-white mb-4">Quick Research Actions</h3>
            <div className="space-y-2">
              <Link
                to="/predict"
                className="w-full flex items-center justify-between p-3 rounded-xl bg-white/3 border border-white/5 text-xs text-white hover:bg-white/5 hover:border-primary-500/30 transition-all group"
              >
                <span className="flex items-center gap-2">
                  <MdScience className="text-primary-400 group-hover:scale-110 transition-transform" />
                  Start New Prediction
                </span>
                <TbArrowRight className="text-white/40 group-hover:translate-x-1 transition-transform" />
              </Link>

              <label className="w-full flex items-center justify-between p-3 rounded-xl bg-white/3 border border-white/5 text-xs text-white hover:bg-white/5 hover:border-primary-500/30 transition-all group cursor-pointer">
                <span className="flex items-center gap-2">
                  <MdCloudUpload className="text-accent-400 group-hover:scale-110 transition-transform" />
                  Upload Molecule File
                </span>
                <input
                  type="file"
                  accept=".csv,.txt,.sdf,.smi"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <TbArrowRight className="text-white/40 group-hover:translate-x-1 transition-transform" />
              </label>

              <Link
                to="/documentation"
                className="w-full flex items-center justify-between p-3 rounded-xl bg-white/3 border border-white/5 text-xs text-white hover:bg-white/5 hover:border-primary-500/30 transition-all group"
              >
                <span className="flex items-center gap-2">
                  <MdMenuBook className="text-violet-400 group-hover:scale-110 transition-transform" />
                  Open Model Documentation
                </span>
                <TbArrowRight className="text-white/40 group-hover:translate-x-1 transition-transform" />
              </Link>

              <Link
                to="/research"
                className="w-full flex items-center justify-between p-3 rounded-xl bg-white/3 border border-white/5 text-xs text-white hover:bg-white/5 hover:border-primary-500/30 transition-all group"
              >
                <span className="flex items-center gap-2">
                  <MdBook className="text-green-400 group-hover:scale-110 transition-transform" />
                  View Research Objective
                </span>
                <TbArrowRight className="text-white/40 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
