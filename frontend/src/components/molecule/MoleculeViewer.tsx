import React, { useRef, useEffect, useCallback } from 'react';
import { TbAtom2 } from 'react-icons/tb';
import { ImportantBond } from '@/types';

interface MoleculeViewerProps {
  smiles: string;
  importantAtoms?: number[];
  importantBonds?: ImportantBond[];
  width?: number;
  height?: number;
  className?: string;
}

interface Atom {
  x: number;
  y: number;
  symbol: string;
  index: number;
  isImportant: boolean;
}

interface Bond {
  from: number;
  to: number;
  isImportant: boolean;
  weight: number;
}

/**
 * MoleculeViewer — Custom Canvas-based 2D molecular structure visualizer.
 *
 * This component generates a deterministic, visually appealing 2D graph
 * representation of a molecular structure from the SMILES string.
 *
 * When the full AI model is integrated:
 * - importantAtoms will receive atom indices with high GNN attention weights
 * - importantBonds will receive bond pairs with attention weights
 * - These will be highlighted with gradient overlays
 *
 * For full cheminformatics rendering, consider integrating:
 * - SmilesDrawer (npm: smiles-drawer)
 * - RDKIT.js (WebAssembly port of RDKit)
 */
export const MoleculeViewer: React.FC<MoleculeViewerProps> = ({
  smiles,
  importantAtoms = [],
  importantBonds = [],
  width = 400,
  height = 300,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const parseSmiles = useCallback((smiles: string): { atoms: Atom[]; bonds: Bond[] } => {
    // Parse atom symbols from SMILES
    const atomSymbols: string[] = [];
    const regex = /([A-Z][a-z]?|\[.*?\])/g;
    let match;
    while ((match = regex.exec(smiles)) !== null) {
      const sym = match[0].replace(/[\[\]]/g, '').replace(/[0-9@+\-H]/g, '');
      if (sym) atomSymbols.push(sym || 'C');
    }

    // Cap atom count for visualization
    const count = Math.min(atomSymbols.length || 6, 16);

    // Place atoms in a deterministic circular/ring layout
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.32;

    const atoms: Atom[] = Array.from({ length: count }, (_, i) => {
      const angle = (2 * Math.PI * i) / count - Math.PI / 2;
      const r = count > 4 ? radius : radius * 0.7;
      return {
        x: cx + r * Math.cos(angle),
        y: cy + r * Math.sin(angle),
        symbol: atomSymbols[i] ?? 'C',
        index: i,
        isImportant: importantAtoms.includes(i),
      };
    });

    // Connect atoms in a ring with optional branches
    const bonds: Bond[] = atoms.map((_atom, i) => {
      const nextIdx = (i + 1) % count;
      const bondPair = importantBonds.find(
        (b) => (b.atomA === i && b.atomB === nextIdx) || (b.atomB === i && b.atomA === nextIdx)
      );
      return {
        from: i,
        to: nextIdx,
        isImportant: !!bondPair,
        weight: bondPair?.weight ?? 0,
      };
    });

    return { atoms, bonds };
  }, [smiles, importantAtoms, importantBonds, width, height]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (!smiles) return;

    const { atoms, bonds } = parseSmiles(smiles);

    // Draw bonds
    bonds.forEach((bond) => {
      const a = atoms[bond.from];
      const b = atoms[bond.to];

      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);

      if (bond.isImportant) {
        const gradient = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
        gradient.addColorStop(0, `rgba(6, 182, 212, ${0.4 + bond.weight * 0.5})`);
        gradient.addColorStop(1, `rgba(59, 130, 246, ${0.4 + bond.weight * 0.5})`);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2.5;
      } else {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.5;
      }
      ctx.stroke();
    });

    // Draw atoms
    atoms.forEach((atom) => {
      // Glow for important atoms
      if (atom.isImportant) {
        const glow = ctx.createRadialGradient(atom.x, atom.y, 0, atom.x, atom.y, 18);
        glow.addColorStop(0, 'rgba(6, 182, 212, 0.35)');
        glow.addColorStop(1, 'rgba(6, 182, 212, 0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(atom.x, atom.y, 18, 0, Math.PI * 2);
        ctx.fill();
      }

      // Atom circle
      const r = atom.symbol.length > 1 ? 14 : 12;
      ctx.beginPath();
      ctx.arc(atom.x, atom.y, r, 0, Math.PI * 2);

      if (atom.isImportant) {
        const grad = ctx.createRadialGradient(atom.x - 3, atom.y - 3, 1, atom.x, atom.y, r);
        grad.addColorStop(0, '#22d3ee');
        grad.addColorStop(1, '#2563eb');
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = 'rgba(30, 41, 59, 0.95)';
      }

      ctx.fill();
      ctx.strokeStyle = atom.isImportant ? 'rgba(6, 182, 212, 0.6)' : 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Atom label
      ctx.fillStyle = atom.isImportant ? '#ffffff' : 'rgba(255,255,255,0.7)';
      ctx.font = `${atom.symbol.length > 1 ? '9px' : '10px'} JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(atom.symbol, atom.x, atom.y);
    });
  }, [parseSmiles, smiles, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  if (!smiles) {
    return (
      <div
        className={`molecule-viewer ${className}`}
        style={{ width, height }}
      >
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
