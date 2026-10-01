"""Narrow orchestration checks; no Git mutation, GPU work or real jobs."""
import json, os, tempfile, unittest
from pathlib import Path
from receiver import Receiver, identity, alive, atomic

class ReceiverChecks(unittest.TestCase):
    def fixture(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);root=Path(d.name)
        config=root/'config.json';atomic(config,{'runtime_root':str(root),'publication_worktree':str(root/'publication'),'bridge':str(root/'bridge')})
        return Receiver(config)
    def test_pause_persists_across_unchanged_control_generation(self):
        r=self.fixture();atomic(r.root/'control.json',{'generation':1,'paused':True})
        self.assertEqual(r.controls(None,None),'pause')
        self.assertEqual(r.controls(None,None),'pause')
        atomic(r.root/'control.json',{'generation':2,'paused':False})
        self.assertIsNone(r.controls(None,None));self.assertIsNone(r.controls(None,None))
    def test_live_identity_and_wrong_start_time(self):
        self.assertTrue(alive({'pid':os.getpid(),'start_ticks':identity(os.getpid())}))
        self.assertFalse(alive({'pid':os.getpid(),'start_ticks':'wrong'}))
        self.assertFalse(alive({'pid':99999999,'start_ticks':'0'}))
    def test_terminal_publication_is_bounded(self):
        from unittest.mock import patch
        r=self.fixture();attempts=[]
        def failure():attempts.append(1);raise RuntimeError('offline')
        r.publish=failure;r.export_files=lambda:None
        with patch('receiver.time.sleep'):r.final_publish()
        self.assertEqual(len(attempts),3);self.assertTrue(r.state['unsynced'])
    def test_stop_does_not_signal_active_job_without_explicit_flag(self):
        from unittest.mock import patch
        r=self.fixture();atomic(r.root/'control.json',{'generation':1,'stop':True})
        with patch('receiver.os.killpg') as kill:
            self.assertEqual(r.controls(None,None),'stop');kill.assert_not_called()
    def test_final_analysis_uses_last_published_exports_and_independent_gate(self):
        r=self.fixture();calls=[]
        r.publish=lambda:calls.append('publish_exports')
        r.collect_other=lambda:calls.append('fetch_peer')
        def aggregate():
            self.assertEqual(calls,['publish_exports','fetch_peer'])
            calls.append('aggregate');r.state['consolidation_status']='complete'
        r.aggregate=aggregate;r.final_publish=lambda:calls.append('publish_analysis')
        r.finalize_results({'all_required_validated':True})
        self.assertEqual(calls,['publish_exports','fetch_peer','aggregate','publish_analysis'])
        self.assertEqual(r.state['phase'],'TERMINAL_PARTIAL')
        atomic(r.root/'native_analysis/analysis_manifest.json',{'status':'complete'})
        calls.clear();r.finalize_results({'all_required_validated':True})
        self.assertEqual(r.state['phase'],'COMPLETE')
if __name__=='__main__':unittest.main(verbosity=2)
