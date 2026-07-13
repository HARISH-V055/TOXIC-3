import mongoose, { Schema, Document, Model } from 'mongoose';

export type PredictionResult = 'toxic' | 'non-toxic' | 'pending' | 'error';

export interface ImportantBond {
  atomA: number;
  atomB: number;
  weight: number;
}

export interface IPrediction extends Document {
  _id: mongoose.Types.ObjectId;
  user: mongoose.Types.ObjectId;
  smiles: string;
  prediction: PredictionResult;
  probability: number | null;
  confidence: number | null;
  importantAtoms: number[];
  importantBonds: ImportantBond[];
  executionTime: number | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface IPredictionModel extends Model<IPrediction> {}

const ImportantBondSchema = new Schema<ImportantBond>(
  {
    atomA: { type: Number, required: true },
    atomB: { type: Number, required: true },
    weight: { type: Number, required: true, min: 0, max: 1 },
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
      enum: ['toxic', 'non-toxic', 'pending', 'error'],
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
    importantAtoms: {
      type: [Number],
      default: [],
    },
    importantBonds: {
      type: [ImportantBondSchema],
      default: [],
    },
    executionTime: {
      type: Number,
      min: 0,
      default: null,
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
