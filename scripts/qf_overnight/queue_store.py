"""Transactional local-only DAG with process identity; never synchronize the DB."""
import json,os,sqlite3,time
from pathlib import Path
from common import read,write,alive,process_start,utc

TERMINAL={'SUCCEEDED','BLOCKED','FAILED','CANCELLED'}
class Queue:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'queue.sqlite',timeout=30)
        self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT,state TEXT,attempt INTEGER DEFAULT 0,retries INTEGER DEFAULT 0,oom INTEGER DEFAULT 0,
 pid INTEGER,start TEXT,heartbeat REAL,progress REAL,not_before REAL DEFAULT 0,detail TEXT,result TEXT,source TEXT);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,time TEXT,job TEXT,event TEXT,detail TEXT);''');self.db.commit()
    def close(self):self.db.close()
    def meta(self,key,default=None):
        row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone();return json.loads(row[0]) if row else default
    def setmeta(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(value)));self.db.commit()
    def event(self,job,event,detail):
        self.db.execute('INSERT INTO events(time,job,event,detail) VALUES(?,?,?,?)',(utc(),job,event,json.dumps(detail)))
    def initialize(self,manifest,owner,source):
        existing=self.meta('protocol_hash')
        if existing:assert existing==manifest['protocol_hash'] and self.meta('owner')==owner
        else:
            self.setmeta('protocol_hash',manifest['protocol_hash']);self.setmeta('owner',owner);self.setmeta('started_at',time.time())
        for job in manifest['jobs']:
            if job['owner']!=owner:continue
            state='WAITING_ADAPTER' if job['kind']=='adapter' else 'PENDING'
            self.db.execute('INSERT OR IGNORE INTO jobs(id,payload,state,source) VALUES(?,?,?,?)',(job['job_id'],json.dumps(job),state,str(source)))
        self.db.commit()
    def rows(self):return [dict(x) for x in self.db.execute('SELECT * FROM jobs ORDER BY id')]
    def get(self,id):return dict(self.db.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone())
    def result(self,id):
        row=self.get(id);assert row['state']=='SUCCEEDED',id
        return read(row['result'])
    def recover(self):
        self.db.execute('BEGIN IMMEDIATE')
        for row in self.db.execute("SELECT * FROM jobs WHERE state='RUNNING'").fetchall():
            if alive(row['pid'],row['start']):continue
            if time.time()-(row['heartbeat'] or 0)<30:continue
            if row['retries']<2:
                self.db.execute("UPDATE jobs SET state='PENDING',retries=retries+1,not_before=?,detail=? WHERE id=?",(time.time()+30,json.dumps({'reason':'dead process, exact epoch-boundary resume when available'}),row['id']))
            else:self.db.execute("UPDATE jobs SET state='FAILED',detail=? WHERE id=?",(json.dumps({'reason':'dead process retry budget exhausted'}),row['id']))
            self.event(row['id'],'dead_process_recovered',{'previous_pid':row['pid'],'previous_start':row['start']})
        self.db.commit()
    def refresh(self):
        rows=self.rows();states={r['id']:r['state'] for r in rows}
        for row in rows:
            if row['state']!='PENDING':continue
            job=json.loads(row['payload'])
            if job['config'].get('allow_terminal_dependencies'):continue
            bad=[d for d in job['deps'] if states.get(d) in {'BLOCKED','FAILED','CANCELLED'}]
            if bad:
                self.db.execute("UPDATE jobs SET state='BLOCKED',detail=? WHERE id=?",(json.dumps({'reason':'dependency failed/blocked','dependencies':bad}),row['id']))
                self.event(row['id'],'blocked_dependency',{'dependencies':bad})
        self.db.commit()
    def claim(self,resource):
        self.recover();self.refresh()
        if self.meta('paused',False) or self.meta('stop',False):return None
        self.db.execute('BEGIN IMMEDIATE');rows=self.rows();states={r['id']:r['state'] for r in rows}
        chosen=None
        for row in sorted(rows,key=lambda r:(json.loads(r['payload'])['priority'],r['id'])):
            job=json.loads(row['payload'])
            if row['state']!='PENDING' or job['resource']!=resource or row['not_before']>time.time():continue
            allowed=TERMINAL if job['config'].get('allow_terminal_dependencies') else {'SUCCEEDED'}
            if all(states.get(d) in allowed for d in job['deps']):chosen=row;break
        if chosen:
            now=time.time();self.db.execute("UPDATE jobs SET state='RUNNING',attempt=attempt+1,pid=?,start=?,heartbeat=?,progress=? WHERE id=?",(os.getpid(),process_start(os.getpid()),now,now,chosen['id']))
            self.event(chosen['id'],'claimed',{'resource':resource,'attempt':chosen['attempt']+1})
        self.db.commit();return self.get(chosen['id']) if chosen else None
    def bind(self,id,pid):
        self.db.execute('UPDATE jobs SET pid=?,start=?,heartbeat=? WHERE id=?',(pid,process_start(pid),time.time(),id));self.db.commit()
    def heartbeat(self,id,progress=False,**detail):
        if progress:self.db.execute('UPDATE jobs SET heartbeat=?,progress=?,detail=? WHERE id=?',(time.time(),time.time(),json.dumps(detail),id))
        else:self.db.execute('UPDATE jobs SET heartbeat=? WHERE id=?',(time.time(),id))
        self.db.commit()
    def success(self,id,path):
        result=read(path);assert result.get('validated'),id
        self.db.execute("UPDATE jobs SET state='SUCCEEDED',result=?,heartbeat=? WHERE id=?",(str(path),time.time(),id));self.event(id,'succeeded',{'result':str(Path(path).relative_to(self.root))});self.db.commit()
    def fail(self,id,error,category='scientific'):
        row=self.get(id);detail={'category':category,'error':str(error)[-3000:]};state='BLOCKED';delay=0
        if category=='oom' and row['oom']==0:
            state='PENDING';delay=60;self.db.execute('UPDATE jobs SET oom=1 WHERE id=?',(id,));detail['action']='restart with microbatch4, accumulation8, same32-day windows'
        elif category=='transient' and row['retries']<2:
            state='PENDING';delay=30*(2**row['retries']);self.db.execute('UPDATE jobs SET retries=retries+1 WHERE id=?',(id,))
        elif category=='transient':state='FAILED'
        self.db.execute('UPDATE jobs SET state=?,detail=?,not_before=?,heartbeat=? WHERE id=?',(state,json.dumps(detail),time.time()+delay,time.time(),id));self.event(id,'attempt_failed',detail);self.db.commit()
    def adapter(self,source=None,reason=None):
        id='adapter_TLOB_ADAPTED';row=self.get(id)
        if source:
            self.db.execute("UPDATE jobs SET state='SUCCEEDED',detail=? WHERE id=?",(json.dumps({'adapter_source':str(source)}),id))
            for r in self.rows():
                if json.loads(r['payload'])['config'].get('model')=='TLOB_ADAPTED' and r['state'] in ['PENDING','BLOCKED']:
                    self.db.execute("UPDATE jobs SET state='PENDING',source=? WHERE id=?",(str(source),r['id']))
            self.event(id,'adapter_ready',{'source':str(source)})
        else:
            self.db.execute("UPDATE jobs SET state='BLOCKED',detail=? WHERE id=?",(json.dumps({'reason':reason}),id));self.event(id,'adapter_blocked',{'reason':reason})
        self.db.commit()
    def snapshot(self):
        rows=self.rows();counts={state:sum(r['state']==state for r in rows) for state in sorted(set(r['state'] for r in rows))}
        output=[]
        for row in rows:
            job=json.loads(row['payload']);item={k:row[k] for k in ['id','state','attempt','retries','oom','pid','start','heartbeat','progress']}
            item.update(kind=job['kind'],model=job['config'].get('model'),priority=job['priority'],detail=json.loads(row['detail'] or '{}'))
            output.append(item)
        return {'run_group':'qf-august-overnight-20261001','owner':self.meta('owner'),'protocol_hash':self.meta('protocol_hash'),'updated_at':utc(),
                'counts':counts,'paused':self.meta('paused',False),'stop':self.meta('stop',False),'jobs':output}
