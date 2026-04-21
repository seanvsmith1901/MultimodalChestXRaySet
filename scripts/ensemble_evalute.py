# this is all I got so far lol
import torch 
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_mlp(model, x_test, y_test):
    print("************************************")
    model.eval()
    with torch.no_grad():
        logits = model(x_test).squeeze(1)
        preds  = torch.sigmoid(logits)
        predicted_class = (preds >= model.probability_flag).float()

    acc = 100 * (predicted_class == y_test).float().mean().item()
    auc = roc_auc_score(y_test.numpy(), preds.numpy())

    # this is what actually matters
    cm = confusion_matrix(y_test.numpy(), predicted_class.numpy())
    tn, fp, fn, tp = cm.ravel()

    for threshold in [0.5, 0.3, 0.2, 0.1, 0.05]:
        predicted_class = (preds >= threshold).float()
        cm = confusion_matrix(y_test.numpy(), predicted_class.numpy())
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(
            f"  t={threshold:.2f} | Sensitivity: {sensitivity:.1%} | Specificity: {specificity:.1%} | FN: {fn} | FP: {fp}")

    print(f"Test Accuracy : {acc:.1f}%")
    print(f"Test AUC-ROC  : {auc:.4f}")
    # print(f"\nConfusion Matrix:")
    # print(f"  True Negatives  (correctly said normal)   : {tn}")
    # print(f"  False Positives (incorrectly flagged)      : {fp}")
    # print(f"  False Negatives (missed abnormal cases)   : {fn}  ← this is the dangerous one")
    # print(f"  True Positives  (correctly caught)         : {tp}")
    # print(f"\nSensitivity (recall on positives): {tp/(tp+fn):.1%}")
    # print(f"Specificity (recall on negatives): {tn/(tn+fp):.1%}")

    # say that missing a sick paitent is twice as bad as a false alarm
    # we can use this to try and skew things certian ways. Right now beta is set to 2.
    # beta = 2
    # for threshold in [0.50, 0.30, 0.20, 0.10, 0.05]:
    #     predicted_class = (preds >= threshold).float().numpy()
    #     f2 = fbeta_score(y_test.numpy(), predicted_class, beta=beta)
    #     print(f"t={threshold:.2f} | F2: {f2:.4f}")