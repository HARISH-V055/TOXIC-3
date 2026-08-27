import React, { useRef, useEffect, useCallback } from 'react';
import { TbAtom2 } from 'react-icons/tb';
import { ImportantAtom, ImportantBond, MolecularGraph } from '@/types';

interface MoleculeViewerProps {
  smiles: string;
  molecularGraph?: MolecularGraph;
  importantAtoms?: ImportantAtom[];
  importantBonds?: ImportantBond[];
  width?: number;
  height?: number;
  className?: string;
}

interface AtomDraw {
  x: number;
  y: number;
  symbol: string;
  index: number;
  isImportant: boolean;
  score: number;
}

interface BondDraw {
  from: number;
  to: number;
  isImportant: boolean;
  score: number;
}

/**
 * MoleculeViewer — RDKit 2D molecular structure renderer.
 *
 * Renders exact RDKit 2D coordinates and true bond topology generated
 * by Python RDKit / PyTorch Geometric from backend responses.
 */
export const MoleculeViewer: React.FC<MoleculeViewerProps> = ({
  smiles,
  molecularGraph,
  importantAtoms = [],
  importantBonds = [],
  width = 400,
  height = 300,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const getAtomImportance = useCallback(
    (index: number): { isImportant: boolean; score: number } => {
      const found = importantAtoms.find((a) => a.index === index);
      if (found) return { isImportant: true, score: found.score };
      return { isImportant: false, score: 0 };
    },
    [importantAtoms]
  );

  const getBondImportance = useCallback(
    (u: number, v: number): { isImportant: boolean; score: number } => {
      const found = importantBonds.find(
        (b) => (b.source === u && b.target === v) || (b.source === v && b.target === u)
      );
      if (found) return { isImportant: true, score: found.score };
      return { isImportant: false, score: 0 };
    },
    [importantBonds]
  );

  const parseGraph = useCallback((): { atoms: AtomDraw[]; bonds: BondDraw[] } => {
    if (molecularGraph && molecularGraph.atoms && molecularGraph.atoms.length > 0) {
      const rawAtoms = molecularGraph.atoms;
      const rawBonds = molecularGraph.bonds || [];

      // Calculate bounds for scaling 2D coordinates
      const xs = rawAtoms.map((a) => a.x);
      const ys = rawAtoms.map((a) => a.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      const padding = 50;
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const scaleX = (width - padding * 2) / rangeX;
      const scaleY = (height - padding * 2) / rangeY;
      const scale = Math.min(scaleX, scaleY, 60);

      const offsetX = (width - rangeX * scale) / 2 - minX * scale;
      const offsetY = (height - rangeY * scale) / 2 - minY * scale;

      const atoms: AtomDraw[] = rawAtoms.map((a) => {
        const imp = getAtomImportance(a.index);
        return {
          x: a.x * scale + offsetX,
          y: height - (a.y * scale + offsetY), // Invert Y for canvas coordinate system
          symbol: a.element,
          index: a.index,
          isImportant: imp.isImportant,
          score: imp.score,
        };
      });

      const bonds: BondDraw[] = rawBonds.map((b) => {
        const imp = getBondImportance(b.source, b.target);
        return {
          from: b.source,
          to: b.target,
          isImportant: imp.isImportant,
          score: imp.score,
        };
      });

      return { atoms, bonds };
    }

    // Fallback: Parse SMILES string linearly (no circular wrap)
    const atomSymbols: string[] = [];
    const regex = /([A-Z][a-z]?|\[.*?\])/g;
    let match;
    while ((match = regex.exec(smiles)) !== null) {
      const sym = match[0].replace(/[\[\]]/g, '').replace(/[0-9@+\-H]/g, '');
      if (sym) atomSymbols.push(sym || 'C');
    }

    const count = atomSymbols.length || 1;
    const cx = width / 2;
    const cy = height / 2;
    const spacing = Math.min(width / (count + 1), 60);
    const startX = cx - ((count - 1) * spacing) / 2;

    const atoms: AtomDraw[] = Array.from({ length: count }, (_, i) => {
      const imp = getAtomImportance(i);
      return {
        x: startX + i * spacing,
        y: cy,
        symbol: atomSymbols[i] ?? 'C',
        index: i,
        isImportant: imp.isImportant,
        score: imp.score,
      };
    });

    const bonds: BondDraw[] = [];
    for (let i = 0; i < count - 1; i++) {
      const imp = getBondImportance(i, i + 1);
      bonds.push({
        from: i,
        to: i + 1,
        isImportant: imp.isImportant,
        score: imp.score,
      });
    }

    return { atoms, bonds };
  }, [molecularGraph, smiles, getAtomImportance, getBondImportance, width, height]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (!smiles) return;

    const { atoms, bonds } = parseGraph();

    // 1. Draw true chemical bonds
    bonds.forEach((bond) => {
      const a = atoms.find((item) => item.index === bond.from);
      const b = atoms.find((item) => item.index === bond.to);

      if (!a || !b) return;

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);

      if (bond.isImportant) {
        const gradient = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        gradient.addColorStop(0, `rgba(6, 182, 212, ${0.5 + bond.score * 0.5})`);
        gradient.addColorStop(1, `rgba(59, 130, 246, ${0.5 + bond.score * 0.5})`);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3.0;
      } else {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1.8;
      }
      ctx.stroke();
    });

    // 2. Draw atoms with GNNExplainer importance highlights
    atoms.forEach((atom) => {
      if (atom.isImportant) {
        const glow = ctx.createRadialGradient(atom.x, atom.y, 0, atom.x, atom.y, 20);
        glow.addColorStop(0, 'rgba(6, 182, 212, 0.45)');
        glow.addColorStop(1, 'rgba(6, 182, 212, 0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(atom.x, atom.y, 20, 0, Math.PI * 2);
        ctx.fill();
      }

      const r = atom.symbol.length > 1 ? 14 : 12;
      ctx.beginPath();
      ctx.arc(atom.x, atom.y, r, 0, Math.PI * 2);

      if (atom.isImportant) {
        const grad = ctx.createRadialGradient(atom.x - 3, atom.y - 3, 1, atom.x, atom.y, r);
        grad.addColorStop(0, '#06b6d4');
        grad.addColorStop(1, '#2563eb');
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = 'rgba(30, 41, 59, 0.95)';
      }

      ctx.fill();
      ctx.strokeStyle = atom.isImportant ? 'rgba(6, 182, 212, 0.8)' : 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = atom.isImportant ? '#ffffff' : 'rgba(255, 255, 255, 0.85)';
      ctx.font = `bold ${atom.symbol.length > 1 ? '10px' : '11px'} JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${atom.symbol}${atom.index}`, atom.x, atom.y);
    });
  }, [parseGraph, smiles, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  if (!smiles) {
    return (
      <div className={`molecule-viewer ${className}`} style={{ width, height }}>
        <div className="text-center text-white/20 px-6">
          <TbAtom2 className="text-5xl mx-auto mb-3 text-primary-500/30" />
          <p className="text-sm">Enter a SMILES string to visualize the molecular structure</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`molecule-viewer ${className}`} style={{ width: '100%', height }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ width: '100%', height: '100%' }}
        className="rounded-2xl"
      />
    </div>
  );
};
