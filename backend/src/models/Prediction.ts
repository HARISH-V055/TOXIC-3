import mongoose, { Schema, Document, Model } from 'mongoose';

export interface ImportantAtom {
  index: number;
  element: string;
  score: number;
}

export interface ImportantBond {
  source: number;
  target: number;
  score: number;
}

export interface GraphAtom {
  index: number;
  element: string;
  x: number;
  y: number;
}

export interface GraphBond {
  source: number;
  target: number;
}

export interface MolecularGraph {
  atoms: GraphAtom[];
  bonds: GraphBond[];
}

export interface IPrediction extends Document {
  _id: mongoose.Types.ObjectId;
  user: mongoose.Types.ObjectId;
  smiles: string;
  prediction: string;
  probability: number | null;
  confidence: number | null;
  threshold: number;
  endpoint: string;
  inferenceTimeMs: number | null;
  importantAtoms: ImportantAtom[];
  importantBonds: ImportantBond[];
  molecularGraph: MolecularGraph;
  explanationImage: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface IPredictionModel extends Model<IPrediction> {}

const ImportantAtomSchema = new Schema<ImportantAtom>(
  {
    index: { type: Number, required: true },
    element: { type: String, required: true },
    score: { type: Number, required: true },
  },
  { _id: false }
);

const ImportantBondSchema = new Schema<ImportantBond>(
  {
    source: { type: Number, required: true },
    target: { type: Number, required: true },
    score: { type: Number, required: true },
  },
  { _id: false }
);

const GraphAtomSchema = new Schema<GraphAtom>(
  {
    index: { type: Number, required: true },
    element: { type: String, required: true },
    x: { type: Number, required: true },
    y: { type: Number, required: true },
  },
  { _id: false }
);

const GraphBondSchema = new Schema<GraphBond>(
  {
    source: { type: Number, required: true },
    target: { type: Number, required: true },
  },
  { _id: false }
);

const MolecularGraphSchema = new Schema<MolecularGraph>(
  {
    atoms: { type: [GraphAtomSchema], default: [] },
    bonds: { type: [GraphBondSchema], default: [] },
  },
  { _id: false }
);

const PredictionSchema = new Schema<IPrediction>(
  {
    user: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: [true, 'User reference is required'],
      index: true,
    },
    smiles: {
      type: String,
      required: [true, 'SMILES string is required'],
      trim: true,
      maxlength: [10000, 'SMILES string is too long'],
    },
    prediction: {
      type: String,
      default: 'pending',
    },
    probability: {
      type: Number,
      min: 0,
      max: 1,
      default: null,
    },
    confidence: {
      type: Number,
      min: 0,
      max: 1,
      default: null,
    },
    threshold: {
      type: Number,
      default: 0.75,
    },
    endpoint: {
      type: String,
      default: 'Tox21 SR-p53',
    },
    inferenceTimeMs: {
      type: Number,
      min: 0,
      default: null,
    },
    importantAtoms: {
      type: [ImportantAtomSchema],
      default: [],
    },
    importantBonds: {
      type: [ImportantBondSchema],
      default: [],
    },
    molecularGraph: {
      type: MolecularGraphSchema,
      default: () => ({ atoms: [], bonds: [] }),
    },
    explanationImage: {
      type: String,
      default: '/outputs/explanations/molecule_explanation.png',
    },
  },
  {
    timestamps: true,
    toJSON: {
      transform: (_doc, ret: Record<string, unknown>) => {
        delete ret.__v;
        return ret;
      },
    },
  }
);

// Compound index for user-based queries with time sorting
PredictionSchema.index({ user: 1, createdAt: -1 });
PredictionSchema.index({ smiles: 'text' }); // Text index for SMILES search

export const Prediction = mongoose.model<IPrediction, IPredictionModel>(
  'Prediction',
  PredictionSchema
);
