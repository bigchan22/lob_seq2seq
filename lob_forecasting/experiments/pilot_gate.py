def decide(ua,um,ux,uc,wins,complete=True):
 if not complete:return 'INCOMPLETE'
 if ua['log_loss']>=um['log_loss']:return 'FAIL'
 class_collapse=(ua['macro_f1']<um['macro_f1']-.001 and ua['mcc']<um['mcc']-.001)
 if ua['log_loss']<ux['log_loss'] and ua['log_loss']<uc['log_loss'] and wins>=14 and not class_collapse:return 'PASS'
 if wins in (12,13) or ua['log_loss']>=ux['log_loss'] or ua['log_loss']>=uc['log_loss'] or class_collapse:return 'MARGINAL'
 return 'INCOMPLETE'
