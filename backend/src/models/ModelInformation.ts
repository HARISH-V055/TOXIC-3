import mongoose, { Schema, Document, Model } from 'mongoose';

export type ModelStatus = 'active' | 'training' | 'offline';

export interface IModelInformation extends Document {
  _id: mongoose.Types.ObjectId;
  version: string;
  name: string;
  description: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1Score: number | null;
  auc: number | null;
  status: ModelStatus;
  lastUpdated: Date;
  trainingDataSize: number | null;
  parameters: number | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface IModelInformationModel extends Model<IModelInformation> {
  getActive(): Promise<IModelInformation | null>;
}

const ModelInformationSchema = new Schema<IModelInformation>(
  {
    version: {
      type: String,
      required: [true, 'Model version is required'],
      unique: true,
      trim: true,
    },
    name: {
      type: String,
      default: 'EQ-KA-GCN',
      trim: true,
    },
    description: {
      type: String,
      default: 'Equivariant Knowledge-Aware Graph Convolutional Network for molecular toxicity prediction',
      trim: true,
    },
    accuracy: { type: Number, min: 0, max: 1, default: null },
    precision: { type: Number, min: 0, max: 1, default: null },
    recall: { type: Number, min: 0, max: 1, default: null },
    f1Score: { type: Number, min: 0, max: 1, default: null },
    auc: { type: Number, min: 0, max: 1, default: null },
    status: {
      type: String,
      enum: ['active', 'training', 'offline'],
      default: 'offline',
    },
    lastUpdated: {
      type: Date,
      default: Date.now,
    },
    trainingDataSize: { type: Number, default: null },
    parameters: { type: Number, default: null },
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

// Static method to get the currently active model
ModelInformationSchema.statics.getActive = function () {
  return this.findOne({ status: 'active' }).sort({ lastUpdated: -1 });
};

export const ModelInformation = mongoose.model<IModelInformation, IModelInformationModel>(
  'ModelInformation',
  ModelInformationSchema
);
