# this is all I got so far lol
import torch 
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.data.dataset import ChestXRayDataset
from src.models.cnn import FinetunedDenseNet
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def final_evaluate_mlp_on_test(model_state_dict, mode, text_amount="medium", device="cpu"):
    """
    Evaluate trained MLP on the HELD-OUT test set.
    Loads the saved vectorizer and model, then evaluates.
    
    Args:
        model_state_dict: State dict of the trained MLP model
        mode: "pneumonia" or "abnormal"
        text_amount: "sparse", "medium", or "full"
        device: "cuda" or "cpu"
    
    Returns:
        Dictionary with metrics: auc, accuracy, sensitivity, specificity, etc.
    """
    import pickle
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from src.data.nlp_preprocessing import build_nlp_pipeline, prepare_text_features
    from src.models.mlp import MLP
    
    SPLITS_DIR = Path("../data/splits")
    MODELS_DIR = Path("../models")
    
    print(f"\n{'='*60}")
    print(f"FINAL MLP TEST EVALUATION - {mode.upper()}")
    print(f"{'='*60}")
    
    # 1. Load the untouched test set
    test_df = pd.read_csv(SPLITS_DIR / f"{mode}_test.csv")
    print(f"Test set: {len(test_df)} samples ({test_df['label'].mean():.1%} pos)")
    
    # 2. Load the saved vectorizer
    vectorizer_path = MODELS_DIR / f"vectorizer_{mode}_{text_amount}.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✅ Loaded vectorizer from: {vectorizer_path}")
    
    # 3. Prepare features
    nlp = build_nlp_pipeline()
    test_df = prepare_text_features(test_df.copy(), nlp, mode=text_amount)
    X_test = vectorizer.transform(test_df["text_clean"]).toarray()
    y_test = test_df["label"].values
    
    print(f"Features: {X_test.shape}")
    
    # 4. Determine input dimension from state dict
    input_dim = model_state_dict['network.0.weight'].shape[1]
    print(f"Model input dimension: {input_dim}")
    
    # 5. Load model
    model = MLP(input_nodes=input_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # 6. Convert to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 7. Evaluate
    metrics = evaluate_mlp(model, X_test_tensor, y_test_tensor)
    
    return metrics


def evaluate_mlp(model, x_test, y_test):
    """
    Evaluate MLP model and print metrics.
    
    Args:
        model: Trained MLP model
        x_test: Test features (torch tensor)
        y_test: Test labels (torch tensor)
    
    Returns:
        Dictionary with metrics
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
    
    print("\n************************************")
    model.eval()
    
    with torch.no_grad():
        logits = model(x_test).squeeze()
        probs = torch.sigmoid(logits)
    
    # Convert to numpy
    probs_np = probs.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    # Threshold sweep
    for threshold in [0.5, 0.3, 0.2, 0.1, 0.05]:
        preds = (probs_np >= threshold).astype(int)
        cm = confusion_matrix(y_test_np, preds)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  t={threshold:.2f} | Sensitivity: {sensitivity:.1%} | Specificity: {specificity:.1%} | FN: {fn} | FP: {fp}")
    
    # Default threshold = 0.5
    preds_default = (probs_np >= 0.5).astype(int)
    cm = confusion_matrix(y_test_np, preds_default)
    tn, fp, fn, tp = cm.ravel()
    
    auc = roc_auc_score(y_test_np, probs_np)
    acc = accuracy_score(y_test_np, preds_default)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTest Accuracy : {acc:.1%}")
    print(f"Test AUC-ROC  : {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correctly said normal)   : {tn}")
    print(f"  False Positives (incorrectly flagged)      : {fp}")
    print(f"  False Negatives (missed abnormal cases)   : {fn}  ← this is the dangerous one")
    print(f"  True Positives  (correctly caught)         : {tp}")
    print(f"\nSensitivity (recall on positives): {sens:.1%}")
    print(f"Specificity (recall on negatives): {spec:.1%}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'probs': probs_np,
        'labels': y_test_np
    }


def final_evaluate_cnn_on_test(model_state_dict, mode, image_dir, device="cuda"):
    """
    Evaluate trained CNN on the HELD-OUT test set.
    Call this ONLY ONCE at the very end.
    """

    SPLITS_DIR = Path("../data/splits")
    
    # Load the untouched test set
    test_df = pd.read_csv(SPLITS_DIR / f"{mode}_test.csv")
    image_size = 224 # ok yeah whatever. 
    
    # Create test dataset
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = ChestXRayDataset(test_df, image_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)
    
    # Load model
    model = FinetunedDenseNet().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"FINAL CNN TEST EVALUATION - {mode.upper()}")
    # print(f"{'='*60}")
    # print(f"Test set: {len(test_df)} samples ({test_df['label'].mean():.1%} pos)")
    
    # Evaluate
    test_metrics = evaluate_cnn(model, test_loader, device)
    
    return test_metrics


def evaluate_cnn(model, test_loader, device="cuda"):
    """
    Evaluate CNN model on a data loader.
    Prints confusion matrix, sensitivity, specificity, AUC, and accuracy.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for evaluation data
        device: "cuda" or "cpu"
    """
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
    import numpy as np
    import torch
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Threshold sweep
    print("\n************************************")
    for t in [0.5, 0.3, 0.2, 0.1, 0.05]:
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  t={t:.2f} | Sensitivity: {sens:.1%} | Specificity: {spec:.1%} | FN: {fn} | FP: {fp}")
    
    # Default threshold = 0.5
    preds = (all_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nTest Accuracy : {acc:.1%}")
    print(f"Test AUC-ROC  : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (correctly said normal)   : {tn}")
    print(f"  False Positives (incorrectly flagged)      : {fp}")
    print(f"  False Negatives (missed abnormal cases)   : {fn}  ← this is the dangerous one")
    print(f"  True Positives  (correctly caught)         : {tp}")
    print(f"\nSensitivity (recall on positives): {sens:.1%}")
    print(f"Specificity (recall on negatives): {spec:.1%}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'probs': all_probs,
        'labels': all_labels
    }