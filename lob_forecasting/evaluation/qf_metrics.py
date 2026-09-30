import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,log_loss,matthews_corrcoef,precision_recall_fscore_support
def multiclass_brier(y,p):return float(np.mean(np.sum((p-np.eye(3)[np.asarray(y)])**2,axis=1)))
def expected_calibration_error(y,p,bins=10):
 conf=p.max(1);pred=p.argmax(1);correct=pred==np.asarray(y);total=len(y);score=0.
 for lo in np.linspace(0,1,bins,endpoint=False):
  mask=(conf>=lo)&(conf<lo+1/bins)
  if mask.any():score+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
 return float(score)
def classification_metrics(y,p):
 pred=np.asarray(p).argmax(1);pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
 return {'log_loss':log_loss(y,p,labels=[0,1,2]),'macro_f1':f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),'mcc':matthews_corrcoef(y,pred),'accuracy':accuracy_score(y,pred),'brier':multiclass_brier(y,p),'ece':expected_calibration_error(y,np.asarray(p)),'precision':pr.tolist(),'recall':re.tolist(),'class_f1':f1.tolist(),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),'prediction_count':len(y)}
