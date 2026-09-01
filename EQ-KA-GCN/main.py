"""
Main Entry Point for EQ-KA-GCN Scientific Project Pipeline

Coordinates initialization, dataset loading, validation, preprocessing, statistics generation,
graph dataset construction, stratified splitting, DataLoader construction, model training,
and baseline GCN model evaluation with automated text/JSON reports and publication-quality figures.
Emits standard IEEE publication project metadata.
"""

import json
import sys
from pathlib import Path
from typing import List
import torch
from torch_geometric.data import Data

from config import get_config
from datasets import load_dataset, validate_dataset, clean_dataset, dataset_statistics
from graph import (
    smiles_to_graph,
    draw_molecule,
    print_graph_info,
    DatasetBuilder,
    GraphDataset,
    compute_and_log_dataset_statistics,
)
from models import BaselineGCN, KAGCN, get_loss_criterion
from training import (
    split_graph_dataset,
    create_dataloaders,
    create_optimizer,
    create_scheduler,
    EarlyStopping,
    History,
    Trainer,
    compute_positive_class_weight,
)
from evaluation import (
    Evaluator,
    ThresholdOptimizer,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    generate_json_report,
    generate_text_report,
)
from quantization import (
    AdaptiveQATManager,
    calculate_model_size_bytes,
    plot_quantization_figures,
)
from explainability import (
    GNNExplainerModule,
    rank_atom_importance,
    visualize_molecule_explanation,
    plot_explainability_figures,
    generate_explanation_report,
)
from utils import set_seed, get_device, setup_logger


def log_split_statistics(name: str, graphs: List[Data], logger) -> None:
    """Computes and logs the toxicity class balance and size for a dataset split."""
    total = len(graphs)
    if total == 0:
        return
    if graphs[0].y is not None and graphs[0].y.numel() > 1:
        num_tasks = graphs[0].y.numel()
        pos_total = sum(int((g.y == 1).sum().item()) for g in graphs if g.y is not None)
        neg_total = sum(int((g.y == 0).sum().item()) for g in graphs if g.y is not None)
        logger.info("==================================================================")
        logger.info(f"SPLIT STATISTICS - {name.upper()} ({num_tasks} TOX21 ASSAY ENDPOINTS)")
        logger.info("==================================================================")
        logger.info(f"Number of Graphs:                {total}")
        logger.info(f"Total Positive Assay Hits:       {pos_total}")
        logger.info(f"Total Negative / Inactive Assays:{neg_total}")
        logger.info("==================================================================")
    else:
        pos = sum(1 for g in graphs if g.y is not None and int(g.y.item() if isinstance(g.y, torch.Tensor) else g.y) == 1)
        neg = sum(1 for g in graphs if g.y is not None and int(g.y.item() if isinstance(g.y, torch.Tensor) else g.y) == 0)
        pos_pct = (pos / total) * 100 if total > 0 else 0.0
        neg_pct = (neg / total) * 100 if total > 0 else 0.0

        logger.info("==================================================================")
        logger.info(f"SPLIT STATISTICS - {name.upper()}")
        logger.info("==================================================================")
        logger.info(f"Number of Graphs:    {total}")
        logger.info(f"Positive Samples:    {pos} ({pos_pct:.2f}%)")
        logger.info(f"Negative Samples:    {neg} ({neg_pct:.2f}%)")
        logger.info("==================================================================")


def run_pipeline() -> None:
    """
    Orchestrates the EQ-KA-GCN pipeline:
      Phase 1: Project Initialization (directories, seed, device, logging)
      Phase 2: Dataset Ingestion, Validation, Cleaning, Statistics, and Saving
      Phase 3: Molecular Graph Construction Demonstration
      Phase 4: Full Dataset Graph Compilation, Serialization, & Statistics
      Phase 5: Stratified Train/Val/Test Dataset Splitting & DataLoader Generation
      Phase 7: Baseline GCN Model Training Loop & Metrics Logging
      Phase 8: Baseline GCN Model Test Set Evaluation, Plots, & Reports
    """
    # ─── PHASE 1: INITIALIZATION ─────────────────────────────────────────────
    # 1. Load configuration
    config = get_config()

    # 2. Dynamic directory creation for safety
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup logger
    logger = setup_logger(config.paths.logs_dir)

    logger.info("==================================================================")
    logger.info("Initializing EQ-KA-GCN Scientific Project Pipeline")
    logger.info("==================================================================")

    # 4. Set random seed
    set_seed(config.training.seed)
    logger.info(f"Global random seed set to: {config.training.seed}")

    # 5. Detect and log device
    try:
        device = get_device(config.device)
        logger.info(f"Detected computing device: {device}")
    except Exception as e:
        logger.error(f"Failed to initialize computing device: {str(e)}")
        sys.exit(1)

    # 6. Log configuration details
    logger.info("--- Configuration System Settings ---")
    logger.info(f"Project Name:           {config.model.name}")
    logger.info(f"Model Save Target:      {config.paths.checkpoints_dir / config.model.save_filename}")
    logger.info(f"Batch Size:             {config.training.batch_size}")
    logger.info(f"Learning Rate:          {config.training.learning_rate}")
    logger.info(f"Training Epochs:        {config.training.epochs}")
    logger.info(f"QAT Enabled:            {config.quantization.qat_enabled} ({config.quantization.bits}-bit)")
    logger.info(f"Model Hidden Dim:       {config.model.hidden_dim}")
    logger.info(f"Model Dropout:          {config.model.dropout}")
    logger.info(f"Model Layers (GCN):     {config.model.num_gcn_layers}")
    logger.info("==================================================================")
    logger.info("Project initialized successfully. Running Phase 2: Data construction.")

    # ─── PHASE 2: DATA LOADING & PREPROCESSING ──────────────────────────────
    logger.info("Starting Phase 2: Dataset Loading and Preprocessing...")
    
    # Define paths
    raw_csv_path = config.paths.raw_dir / config.data.raw_filename
    processed_csv_path = config.paths.processed_dir / config.data.processed_filename
    
    # 1. Load Dataset
    try:
        df_raw = load_dataset(raw_csv_path)
    except Exception as e:
        logger.error(f"Pipeline terminated. Failed to load raw dataset: {str(e)}")
        sys.exit(1)
        
    # 2. Validate Dataset
    try:
        validate_dataset(
            df_raw, 
            smiles_column=config.data.smiles_column, 
            target_column=config.data.target_columns
        )
    except Exception as e:
        logger.error(f"Pipeline terminated. Dataset validation failed: {str(e)}")
        sys.exit(1)
        
    # 3. Clean Dataset
    df_clean = clean_dataset(
        df_raw, 
        smiles_column=config.data.smiles_column, 
        target_column=config.data.target_columns
    )
    
    # 4. Generate Statistics
    _ = dataset_statistics(
        raw_df=df_raw,
        clean_df=df_clean,
        dataset_name=config.model.name,
        target_column=config.data.target_column,
        smiles_column=config.data.smiles_column
    )
    
    # 5. Save Clean Dataset
    try:
        logger.info(f"Saving cleaned dataset to: {processed_csv_path}")
        df_clean.to_csv(processed_csv_path, index=False)
        logger.info("Cleaned dataset saved successfully.")
    except Exception as e:
        logger.error(f"Pipeline terminated. Failed to save cleaned dataset: {str(e)}")
        sys.exit(1)
        
    logger.info("==================================================================")
    logger.info("Phase 2 complete. Running Phase 3: Graph Construction Demo.")
    logger.info("==================================================================")

    # ─── PHASE 3: GRAPH CONSTRUCTION & VISUALIZATION DEMONSTRATION ─────────
    logger.info("Starting Phase 3: Graph Construction Demonstration...")

    if df_clean.empty:
        logger.error("Cleaned dataset is empty. Cannot demonstrate Phase 3.")
        sys.exit(1)

    # 1. Load one sample molecule from the cleaned dataset
    sample_row = df_clean.iloc[0]
    sample_smiles = sample_row[config.data.smiles_column]
    sample_label = [float(sample_row[c]) for c in config.data.target_columns]

    logger.info(f"Loaded sample molecule SMILES: '{sample_smiles}' with {len(sample_label)} endpoints")

    # 2. Convert it into a graph
    graph_data = smiles_to_graph(sample_smiles, sample_label)

    if graph_data is not None:
        # 3. Print graph information
        print_graph_info(graph_data)

        # 4. Draw the molecule
        img = draw_molecule(sample_smiles)
        if img is not None:
            save_img_path = config.paths.figures_dir / "sample_molecule.png"
            img.save(save_img_path)
            logger.info(f"Successfully drew molecule and saved to: {save_img_path}")
        else:
            logger.warning("Failed to render molecular drawing.")
    else:
        logger.error("Failed to build graph from sample molecule.")

    logger.info("==================================================================")
    logger.info("Phase 3 complete. Running Phase 4: Full Dataset Graph Construction.")
    logger.info("==================================================================")

    # ─── PHASE 4: FULL DATASET GRAPH COMPILATION & SERIALIZATION ───────────
    logger.info("Starting Phase 4: Full Dataset Graph Processing...")

    # Define paths
    graphs_pt_path = config.paths.processed_dir / config.data.graphs_filename
    info_json_path = config.paths.processed_dir / config.data.info_filename

    # 1. Generate graph dataset across all 12 Tox21 endpoints
    builder = DatasetBuilder()
    graphs = builder.build_dataset(
        csv_path=processed_csv_path,
        smiles_column=config.data.smiles_column,
        target_column=config.data.target_columns,
    )

    # 2. Save graphs.pt
    builder.save_dataset(graphs, graphs_pt_path)

    # 3. Reload graphs.pt using GraphDataset wrapper
    logger.info(f"Reloading graph dataset from: {graphs_pt_path}")
    graph_dataset = GraphDataset(graphs_path=graphs_pt_path)
    logger.info(f"Loaded Graph Dataset. Graphs count: {len(graph_dataset)}")

    # 4. Run statistics
    stats = compute_and_log_dataset_statistics(
        dataset=graph_dataset,
        skipped_count=builder.skipped_count,
        dataset_name="Tox21",
        target_column=config.data.target_column,
    )

    # 5. Save stats to dataset_info.json
    try:
        logger.info(f"Writing dataset information metadata to: {info_json_path}")
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("dataset_info.json updated successfully.")
    except Exception as e:
        logger.error(f"Failed to write dataset_info.json metadata: {str(e)}")

    # 6. Print one sample graph from reloaded dataset
    if len(graph_dataset) > 0:
        logger.info("Displaying one sample graph from the reloaded GraphDataset:")
        sample_graph = graph_dataset[0]
        if isinstance(sample_graph, Data):
            print_graph_info(sample_graph)

    logger.info("==================================================================")
    logger.info("Phase 4 complete. Running Phase 5: Stratified Dataset Splitting.")
    logger.info("==================================================================")

    # ─── PHASE 5: STRATIFIED SPLITTING & DATALOADER GENERATION ──────────────
    logger.info("Starting Phase 5: Stratified Dataset Splitting...")

    # Define filenames
    train_graphs_path = config.paths.processed_dir / config.data.train_graphs_filename
    val_graphs_path = config.paths.processed_dir / config.data.val_graphs_filename
    test_graphs_path = config.paths.processed_dir / config.data.test_graphs_filename

    # 1. Split graph list
    train_graphs, val_graphs, test_graphs = split_graph_dataset(
        graphs=graph_dataset.graphs,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=config.training.seed,
    )

    # 2. Save split datasets to disk
    try:
        logger.info(f"Saving train graphs split ({len(train_graphs)} items) to: {train_graphs_path}")
        torch.save(train_graphs, train_graphs_path)

        logger.info(f"Saving validation graphs split ({len(val_graphs)} items) to: {val_graphs_path}")
        torch.save(val_graphs, val_graphs_path)

        logger.info(f"Saving test graphs split ({len(test_graphs)} items) to: {test_graphs_path}")
        torch.save(test_graphs, test_graphs_path)

        logger.info("All stratified splits saved successfully to disk.")
    except Exception as e:
        logger.error(f"Failed to serialize split datasets: {str(e)}")
        sys.exit(1)

    # 3. Create PyTorch Geometric DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        batch_size=config.training.batch_size,
    )

    log_split_statistics("Test Split", test_graphs, logger)

    logger.info("==================================================================")
    logger.info("Phase 5 complete. Running Phase 7 / Phase 10: Model Training.")
    logger.info("==================================================================")

    # ─── PHASE 7 / PHASE 11: MODEL SELECTION & LOSS CRITERION SETUP ──────────
    if config.fourier_kan.enabled:
        if config.fourier_kan.use_focal_loss:
            # ── FOCAL LOSS TRAINING (Primary: Best for severe 18:1 imbalance) ──
            model_name = "Focal KA-GCN"
            best_model_path = config.paths.checkpoints_dir / config.fourier_kan.focal_save_filename
            history_csv_path = config.paths.outputs_dir / config.fourier_kan.focal_history_filename
            json_report_path = config.paths.outputs_dir / "focal_evaluation_report.json"
            text_report_path = config.paths.outputs_dir / "focal_classification_report.txt"
            prefix = "focal_"

            # Compute class balance statistics for logging (not used for criterion)
            pos_weight_ref, pos_count, neg_count = compute_positive_class_weight(train_graphs)

            print("=================================================")
            print("CLASS IMBALANCE ANALYSIS — FOCAL LOSS TRAINING")
            print("=================================================")
            print(f"Positive Samples (Toxic)    : {pos_count}")
            print(f"Negative Samples (Non-Toxic): {neg_count}")
            print(f"Imbalance Ratio             : {pos_weight_ref:.2f}:1 (Neg:Pos)")
            print(f"Loss Function               : Focal Loss")
            print(f"Focal Alpha (pos weight)    : {config.fourier_kan.focal_alpha}")
            print(f"Focal Gamma (focus param)   : {config.fourier_kan.focal_gamma}")
            print("=================================================")

            logger.info(
                f"Using Focal Loss | alpha={config.fourier_kan.focal_alpha}, "
                f"gamma={config.fourier_kan.focal_gamma} | "
                f"Class ratio {neg_count}:{pos_count} (Neg:Pos = {pos_weight_ref:.1f}:1)"
            )

            criterion = get_loss_criterion(
                use_focal_loss=True,
                focal_alpha=config.fourier_kan.focal_alpha,
                focal_gamma=config.fourier_kan.focal_gamma,
            )

        elif config.fourier_kan.use_weighted_loss:
            # ── WEIGHTED BCE TRAINING (Fallback) ──────────────────────────────
            model_name = "Weighted KA-GCN"
            best_model_path = config.paths.checkpoints_dir / config.fourier_kan.weighted_save_filename
            history_csv_path = config.paths.outputs_dir / config.fourier_kan.weighted_history_filename
            json_report_path = config.paths.outputs_dir / "weighted_evaluation_report.json"
            text_report_path = config.paths.outputs_dir / "weighted_classification_report.txt"
            prefix = "ka_gcn_weighted_"

            pos_weight, pos_count, neg_count = compute_positive_class_weight(train_graphs)

            print("=================================================")
            print("CLASS IMBALANCE ANALYSIS — WEIGHTED LOSS TRAINING")
            print("=================================================")
            print(f"Positive Samples         : {pos_count}")
            print(f"Negative Samples         : {neg_count}")
            print(f"Computed Positive Weight : {pos_weight:.4f}")
            print("=================================================")

            criterion = get_loss_criterion(positive_class_weight=pos_weight)

        else:
            # ── STANDARD (UNWEIGHTED) BCE ─────────────────────────────────────
            model_name = "KA-GCN"
            best_model_path = config.paths.checkpoints_dir / config.fourier_kan.save_filename
            history_csv_path = config.paths.outputs_dir / config.fourier_kan.history_filename
            json_report_path = config.paths.outputs_dir / "ka_gcn_evaluation_report.json"
            text_report_path = config.paths.outputs_dir / "ka_gcn_classification_report.txt"
            prefix = "ka_gcn_"
            criterion = get_loss_criterion(positive_class_weight=None)

        logger.info(f"FourierKAN configuration ENABLED ({model_name}). Instantiating KAGCN model.")
        model = KAGCN(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            fp_dim=config.model.fp_dim if config.model.use_fingerprint else 0,
            gcn_dropout=config.model.dropout,
            kan_hidden_dim=config.fourier_kan.hidden_dim,
            fourier_order=config.fourier_kan.fourier_order,
            kan_dropout=config.fourier_kan.dropout,
            kan_activation=config.fourier_kan.activation,
        )
    else:
        model_name = "Baseline GCN"
        best_model_path = config.paths.checkpoints_dir / config.model.save_filename
        history_csv_path = config.paths.outputs_dir / "history.csv"
        json_report_path = config.paths.outputs_dir / "evaluation_report.json"
        text_report_path = config.paths.outputs_dir / "classification_report.txt"
        prefix = ""
        criterion = get_loss_criterion(positive_class_weight=None)

        logger.info("FourierKAN configuration DISABLED. Instantiating BaselineGCN model.")
        model = BaselineGCN(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            dropout=config.model.dropout,
        )

    model.to(device)
    logger.info(f"Model ({model_name}) Architecture:\n{str(model)}")

    # 1. Set Up Training Utilities & Checkpoint Reuse
    if best_model_path.exists():
        logger.info(f"Existing trained checkpoint found at '{best_model_path}'. Reusing trained weights without retraining per Phase 12 specification.")
    else:
        optimizer = create_optimizer(
            model=model,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = create_scheduler(
            optimizer=optimizer,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        early_stopping = EarlyStopping(
            patience=config.training.early_stopping,
            save_path=str(best_model_path),
        )

        history = History()

        trainer = Trainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            history=history,
        )

        logger.info(f"Starting {model_name} Training Loop...")
        best_val_loss, best_epoch, training_time = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
        )

        history.export_csv(str(history_csv_path))

        print("=================================================")
        print(f"{model_name.upper()} TRAINING COMPLETE")
        print("=================================================")
        print(f"Best Validation Loss : {best_val_loss:.6f}")
        print(f"Best Epoch           : {best_epoch}")
        print(f"Training Time        : {training_time:.2f} seconds")
        print(f"Best Checkpoint Path : {best_model_path}")
        print("=================================================")

    # ─── PHASE 12: THRESHOLD OPTIMIZATION & EVALUATION ───────────────────────
    logger.info(f"Starting Model Evaluation for {model_name} on Held-Out Test Set...")
    evaluator = Evaluator(model=model, device=device)
    evaluator.load_model(str(best_model_path))

    outputs_figures_dir = config.paths.outputs_dir / "figures"
    outputs_figures_dir.mkdir(parents=True, exist_ok=True)

    if config.threshold.enabled:
        logger.info("==================================================================")
        logger.info("PHASE 12: ADAPTIVE DECISION THRESHOLD OPTIMIZATION")
        logger.info("==================================================================")

        # 1. Evaluate on Validation Set to extract validation probabilities
        logger.info("Running inference on Validation Set for threshold search...")
        _, val_y_true, _, val_y_prob = evaluator.evaluate(
            loader=val_loader, threshold=0.50
        )

        # 2. Run Threshold Grid Search
        thresh_opt = ThresholdOptimizer(
            search_start=config.threshold.search_start,
            search_end=config.threshold.search_end,
            step=config.threshold.step,
            selection_metric=config.threshold.selection_metric,
        )
        optimal_threshold, opt_val_metrics, df_thresh = thresh_opt.grid_search(
            val_y_true, val_y_prob
        )

        # 3. Export CSV Analysis and 300 DPI Figures
        analysis_csv_path = config.paths.outputs_dir / "threshold_analysis.csv"
        thresh_opt.save_csv(df_thresh, str(analysis_csv_path))
        thresh_opt.plot_curves(df_thresh, str(outputs_figures_dir))

        # 4. Evaluate Test Set at default 0.50 threshold (for baseline comparison)
        test_metrics_050, _, _, _ = evaluator.evaluate(
            loader=test_loader, threshold=0.50
        )

        # 5. Evaluate Test Set at Optimal Threshold
        test_metrics, y_true, y_pred, y_prob = evaluator.evaluate(
            loader=test_loader, threshold=optimal_threshold
        )
        test_metrics["optimal_threshold"] = optimal_threshold
        test_metrics["optimized_metrics"] = opt_val_metrics
    else:
        optimal_threshold = 0.50
        test_metrics, y_true, y_pred, y_prob = evaluator.evaluate(
            loader=test_loader, threshold=0.50
        )
        test_metrics_050 = test_metrics

    # 6. Save Reports
    if config.evaluation.save_reports:
        generate_json_report(
            metrics=test_metrics,
            save_path=str(json_report_path),
            model_name=model_name,
            dataset_name="Tox21",
        )
        generate_text_report(
            metrics=test_metrics,
            save_path=str(text_report_path),
        )

    # 7. Save Plots
    if config.evaluation.save_plots:
        plot_roc_curve(y_true, y_prob, str(outputs_figures_dir / f"{prefix}roc_curve.png"))
        plot_precision_recall_curve(y_true, y_prob, str(outputs_figures_dir / f"{prefix}precision_recall_curve.png"))
        cm_data = test_metrics.get("confusion_matrix", None)
        if cm_data is not None:
            plot_confusion_matrix(cm_data, str(outputs_figures_dir / f"{prefix}confusion_matrix.png"))
        if history_csv_path.exists():
            plot_loss_curve(str(history_csv_path), str(outputs_figures_dir / f"{prefix}loss_curve.png"))
            plot_accuracy_curve(str(history_csv_path), str(outputs_figures_dir / f"{prefix}accuracy_curve.png"))

    # 8. Print Summary Block
    if config.threshold.enabled:
        print("=================================================")
        print("THRESHOLD OPTIMIZATION COMPLETE (MULTI-TASK TOX21)")
        print("=================================================")
        print(f"Optimal Threshold   : {optimal_threshold:.2f}")
        print(f"Accuracy            : {test_metrics.get('accuracy', 0.0) * 100:.2f} %")
        print(f"Precision           : {test_metrics.get('precision', 0.0) * 100:.2f} %")
        print(f"Recall              : {test_metrics.get('recall', 0.0) * 100:.2f} %")
        print(f"F1 Score            : {test_metrics.get('f1_score', 0.0) * 100:.2f} %")
        print(f"Balanced Accuracy   : {test_metrics.get('balanced_accuracy', 0.0) * 100:.2f} %")
        print(f"MCC                 : {test_metrics.get('mcc', 0.0):.4f}")
        print(f"ROC-AUC             : {test_metrics.get('roc_auc', 0.0):.4f}")
        print("=================================================")

    # 9. Print Comparison Table
    baseline_report_path = config.paths.outputs_dir / "evaluation_report.json"
    baseline_prec, baseline_rec, baseline_f1 = 0.0, 0.0, 0.0
    if baseline_report_path.exists():
        try:
            with open(baseline_report_path, "r", encoding="utf-8") as bf:
                b_data = json.load(bf)
            baseline_prec = b_data.get("precision", 0.0) * 100
            baseline_rec = b_data.get("recall", 0.0) * 100
            baseline_f1 = b_data.get("f1_score", 0.0) * 100
        except Exception:
            pass

    w_050_prec = test_metrics_050.get("precision", 0.0) * 100
    w_050_rec = test_metrics_050.get("recall", 0.0) * 100
    w_050_f1 = test_metrics_050.get("f1_score", 0.0) * 100

    w_opt_prec = test_metrics.get("precision", 0.0) * 100
    w_opt_rec = test_metrics.get("recall", 0.0) * 100
    w_opt_f1 = test_metrics.get("f1_score", 0.0) * 100

    print("\n-------------------------------------------------------------")
    print(f"{'Model':<22} {'Threshold':<12} {'Precision':<11} {'Recall':<9} {'F1':<8}")
    print("-------------------------------------------------------------")
    print(f"{'Baseline GCN':<22} {'0.50':<12} {baseline_prec:.2f}%      {baseline_rec:.2f}%    {baseline_f1:.2f}%")
    print(f"{'MultiTask KA-GCN (0.50)':<22} {'0.50':<12} {w_050_prec:.2f}%      {w_050_rec:.2f}%    {w_050_f1:.2f}%")
    print(f"{'MultiTask KA-GCN (Opt)':<22} {f'{optimal_threshold:.2f}':<12} {w_opt_prec:.2f}%      {w_opt_rec:.2f}%    {w_opt_f1:.2f}%")
    print("-------------------------------------------------------------")

    # ─── PHASE 13: ADAPTIVE LAYER-WISE QAT ────────────────────────────────────
    if config.quantization.enabled:
        logger.info("==================================================================")
        logger.info("PHASE 13: ADAPTIVE LAYER-WISE QUANTIZATION-AWARE TRAINING (QAT)")
        logger.info("==================================================================")

        # 1. Measure FP32 Latency
        fp32_latency_ms = test_metrics["inference_time_per_sample_ms"]

        # 2. Instantiate AdaptiveQATManager and run calibration
        qat_manager = AdaptiveQATManager(
            target_layers=["conv1", "conv2", "fourier_kan", "fc_out"],
            supported_bits=config.quantization.supported_bits,
        )
        bit_assignments = qat_manager.calibrate_and_assign_bits(
            model=model,
            dataloader=train_loader,
            device=device,
            calibration_batches=config.quantization.calibration_batches,
        )

        # 3. Prepare QAT Model
        qat_model = qat_manager.prepare_qat_model(model, bit_assignments)

        # 4. Set Up QAT Training Utilities
        qat_checkpoint_path = config.paths.checkpoints_dir / config.quantization.qat_save_filename
        qat_history_path = config.paths.outputs_dir / config.quantization.qat_history_filename

        qat_optimizer = create_optimizer(
            model=qat_model,
            lr=config.training.learning_rate * 0.5,
            weight_decay=config.training.weight_decay,
        )
        qat_scheduler = create_scheduler(
            optimizer=qat_optimizer,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        qat_early_stopping = EarlyStopping(
            patience=config.training.early_stopping,
            save_path=str(qat_checkpoint_path),
        )
        qat_history = History()

        qat_trainer = Trainer(
            model=qat_model,
            device=device,
            criterion=criterion,
            optimizer=qat_optimizer,
            scheduler=qat_scheduler,
            early_stopping=qat_early_stopping,
            history=qat_history,
        )

        logger.info("Starting Adaptive QAT fine-tuning loop...")
        qat_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
        )
        qat_history.export_csv(str(qat_history_path))

        # 5. Compute Model Size & Profiling
        fp32_size_kb, quant_size_kb, compression_ratio, memory_reduction = calculate_model_size_bytes(
            qat_model, bit_assignments
        )

        # 6. Evaluate QAT Model on Test Set using Phase 12 Optimal Threshold
        qat_evaluator = Evaluator(model=qat_model, device=device)
        if qat_checkpoint_path.exists():
            qat_evaluator.load_model(str(qat_checkpoint_path))

        qat_test_metrics, _, _, _ = qat_evaluator.evaluate(
            loader=test_loader, threshold=optimal_threshold
        )

        # 7. Generate Publication Figures (300 DPI)
        plot_quantization_figures(
            bit_assignments=bit_assignments,
            fp32_size_kb=fp32_size_kb,
            quant_size_kb=quant_size_kb,
            fp32_latency_ms=fp32_latency_ms,
            quant_latency_ms=qat_test_metrics["inference_time_per_sample_ms"],
            output_dir=str(outputs_figures_dir),
        )

        # 8. Export Quantization JSON Report
        qat_report_data = {
            "model_name": "Adaptive QAT KA-GCN",
            "optimal_threshold": optimal_threshold,
            "layer_bit_assignments": bit_assignments,
            "fp32_size_kb": fp32_size_kb,
            "quantized_size_kb": quant_size_kb,
            "compression_ratio": compression_ratio,
            "memory_reduction_percent": memory_reduction,
            "fp32_inference_time_ms": fp32_latency_ms,
            "quantized_inference_time_ms": qat_test_metrics["inference_time_per_sample_ms"],
            "accuracy": qat_test_metrics.get("accuracy", 0.0),
            "balanced_accuracy": qat_test_metrics.get("balanced_accuracy", 0.0),
            "precision": qat_test_metrics.get("precision", 0.0),
            "recall": qat_test_metrics.get("recall", 0.0),
            "f1_score": qat_test_metrics.get("f1_score", 0.0),
            "roc_auc": qat_test_metrics.get("roc_auc", 0.0),
            "mcc": qat_test_metrics.get("mcc", 0.0),
        }
        qat_report_path = config.paths.outputs_dir / config.quantization.qat_report_filename
        qat_manager.export_quantization_report(qat_report_data, str(qat_report_path))

        # 9. Print Console Summary Block (Requirement 11)
        bit_alloc_str = ", ".join([f"{k}: {v}-bit" for k, v in bit_assignments.items()])
        print("\n=================================================")
        print("ADAPTIVE QAT COMPLETE")
        print("=================================================")
        print(f"Layer-wise Bit Allocation : {bit_alloc_str}")
        print(f"Model Size Before         : {fp32_size_kb:.2f} KB")
        print(f"Model Size After          : {quant_size_kb:.2f} KB")
        print(f"Compression Ratio         : {compression_ratio:.2f} x")
        print(f"Memory Reduction          : {memory_reduction:.2f} %")
        print(f"Inference Time            : {qat_test_metrics.get('inference_time_per_sample_ms', 0.0):.2f} ms/sample")
        print(f"Optimal Threshold         : {optimal_threshold:.2f}")
        print(f"Accuracy                  : {qat_test_metrics.get('accuracy', 0.0) * 100:.2f} %")
        print(f"Precision                 : {qat_test_metrics.get('precision', 0.0) * 100:.2f} %")
        print(f"Recall                    : {qat_test_metrics.get('recall', 0.0) * 100:.2f} %")
        print(f"F1 Score                  : {qat_test_metrics.get('f1_score', 0.0) * 100:.2f} %")
        print(f"Balanced Accuracy         : {qat_test_metrics.get('balanced_accuracy', 0.0) * 100:.2f} %")
        print(f"MCC                       : {qat_test_metrics.get('mcc', 0.0):.4f}")
        print(f"ROC-AUC                   : {qat_test_metrics.get('roc_auc', 0.0):.4f}")
        print("=================================================")

    # ─── PHASE 14: EXPLAINABILITY MODULE (GNNEXPLAINER) ──────────────────────
    if config.explainability.enabled:
        logger.info("==================================================================")
        logger.info("PHASE 14: EXPLAINABILITY MODULE (GNNEXPLAINER)")
        logger.info("==================================================================")

        # 1. Select a representative sample graph from test dataset
        target_model = qat_model if (config.quantization.enabled and 'qat_model' in locals()) else model
        sample_graph = test_loader.dataset[0]

        # Extract SMILES if stored or default representative
        smiles_sample = getattr(sample_graph, "smiles", "CC(=O)Oc1ccccc1C(=O)O")

        # 2. Run GNNExplainer
        explainer = GNNExplainerModule(epochs=100, lr=0.01)
        node_importance, edge_importance = explainer.explain_graph(
            model=target_model,
            graph=sample_graph,
            device=device,
        )

        # 3. Rank Atoms and Bonds
        top_atoms, top_bonds = rank_atom_importance(
            graph=sample_graph,
            node_importance=node_importance,
            edge_importance=edge_importance,
            top_k_atoms=config.explainability.top_k_atoms,
            top_k_bonds=config.explainability.top_k_bonds,
        )

        # 4. Generate Visualizations & 300 DPI Publication Plots
        explanations_dir = config.paths.outputs_dir / "explanations"
        explanations_dir.mkdir(parents=True, exist_ok=True)
        mol_vis_path = explanations_dir / "molecule_explanation.png"

        visualize_molecule_explanation(
            smiles=smiles_sample,
            node_importance=node_importance,
            edge_importance=edge_importance,
            save_path=str(mol_vis_path),
        )

        plot_explainability_figures(
            graph=sample_graph,
            node_importance=node_importance,
            edge_importance=edge_importance,
            top_atoms=top_atoms,
            top_bonds=top_bonds,
            output_dir=str(outputs_figures_dir),
        )

        # 5. Get Sample Prediction & Confidence
        sample_batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long, device=device)
        with torch.no_grad():
            s_logits = target_model(
                x=sample_graph.x.to(device).float(),
                edge_index=sample_graph.edge_index.to(device),
                batch=sample_batch,
                return_logits=True,
            )
            s_probs = torch.sigmoid(s_logits).squeeze()
            if s_probs.numel() > 1:
                s_prob = float(s_probs[-1].item())  # Primary endpoint: SR-p53
            else:
                s_prob = float(s_probs.item())
        sample_pred = "Toxic" if s_prob >= optimal_threshold else "Non-Toxic"
        sample_conf = s_prob if sample_pred == "Toxic" else (1.0 - s_prob)

        # 6. Generate JSON Report
        json_report_path = config.paths.outputs_dir / "explanation_report.json"
        generate_explanation_report(
            smiles=smiles_sample,
            prediction=sample_pred,
            confidence=sample_conf,
            top_atoms=top_atoms,
            top_bonds=top_bonds,
            node_importance=node_importance,
            edge_importance=edge_importance,
            inference_time_ms=0.0,
            save_path=str(json_report_path),
        )

        # 7. Print Console Summary (Requirement 11)
        print("\n=================================================")
        print("EXPLAINABILITY COMPLETE")
        print("=================================================")
        print(f"Prediction          : {sample_pred}")
        print(f"Confidence          : {sample_conf * 100:.2f} %")
        print(f"Top Important Atoms :")
        for atom in top_atoms[:3]:
            print(f"  - {atom['atom_name']} (Index {atom['atom_index']}) : Score {atom['importance_score']:.4f}")
        print(f"Top Important Bonds :")
        for bond in top_bonds[:3]:
            print(f"  - {bond['bond_name']} : Score {bond['importance_score']:.4f}")
        print(f"Visualization Saved : {mol_vis_path}")
        print(f"JSON Report Saved   : {json_report_path}")
        print("=================================================")


if __name__ == "__main__":
    run_pipeline()
