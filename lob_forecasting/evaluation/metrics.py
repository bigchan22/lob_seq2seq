def classification_metrics(y_true, y_pred):
    """Legacy three-class accuracy and fixed-label macro-F1."""
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": accuracy_score(y_true,y_pred), "macro_f1": f1_score(y_true,y_pred,labels=[0,1,2],average="macro",zero_division=0)}
