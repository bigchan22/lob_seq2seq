"""Shared immutable identifiers and safe local serialization; Python 3.9+."""
import datetime, hashlib, json, os, subprocess, tempfile, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
GROUP = 'qf-august-overnight-20261001'
CONFIG = ROOT / 'configs/qf_overnight/20261001'
DOCS = ROOT / 'docs/qf_overnight/20261001'
READY = ROOT / 'coordination/qf_overnight/20261001/ready.json'
HANDOFF = 'e85a89fd56584b513be940e6259c637369bd830b'
REMOTE = 'git@github.com:bigchan22/lob_seq2seq.git'
FAMILIES = ['U1','UC','UA','UM_V2_DEPTH','UX','GRU_U1','SHARED_QUERY','U0','TLOB_ADAPTED']
OWNERS = {m: ('rtx3090' if m in ['UA','UC','SHARED_QUERY'] else 'a5000') for m in FAMILIES}
LRS = [0.00005,0.0001,0.0002,0.0004]
SEEDS = [42,7,123]
POPS = ['P0_legacy','P1_valid_cache_endpoints','P2_candidate_clock','P3_intersection']
PROBS = ['probability_down','probability_flat','probability_up']
KEYS = ['asset_id','date','origin_time','split','fold']
BASE = dict(d_model=128,nhead=8,temporal_layers=2,cross_asset_layers=1,dropout=0.1,
            weight_decay=0.0001,max_epochs=200,min_epochs=20,patience=30,min_delta=0.0001,
            effective_days=32,microbatch=8,gradient_clip_norm=1.0,precision='float32',scheduler=None)
SPLIT_INDICES = {'main': [0,345,394,493], 'R1':[0,345,394,427],
                 'R2':[0,378,427,460], 'R3':[0,411,460,493]}

def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def code_sha():return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
def source_files(optional=False):
    files=sorted(p for p in (ROOT/'scripts/qf_overnight').glob('*.py') if p.name!='tlob_adapter.py')
    files += [ROOT/'lob_forecasting/models/qf_variants.py',ROOT/'lob_forecasting/data/transforms.py',
              ROOT/'lob_forecasting/experiments/discrimination_factors.py',ROOT/'scripts/qf_data_review/build_data_review.py']
    if optional and (ROOT/'scripts/qf_overnight/tlob_adapter.py').exists():
        files.append(ROOT/'scripts/qf_overnight/tlob_adapter.py')
        files += [p for p in (ROOT/'scripts/qf_overnight/vendor_tlob').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
def process_start(pid):
    try:return Path('/proc/%d/stat'%int(pid)).read_text().rsplit(')',1)[1].split()[19]
    except (OSError,ValueError,IndexError):return None
def alive(pid,start):return bool(pid and start and process_start(pid)==str(start))
def lr_tag(lr):return '%06d'%round(lr*1e7)
def fit_name(model,seed,lr=None,fold='main',asset=None):
    return '_'.join([fold,model,str(seed),lr_tag(lr) if lr is not None else 'selected']+([asset] if asset else []))
