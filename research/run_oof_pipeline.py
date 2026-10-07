"""Resumable training/evaluation pipeline; each stage records explicit completion."""
import json,subprocess,sys,time
from oof_common import RUN,PROJECT,SUBJECTS,save

def phase(name,**extra):save(RUN/'status.json',dict(stage=name,complete=False,**extra));print(name,flush=True)
def run(script,log):
    with (RUN/log).open('w') as output:subprocess.run([sys.executable,str(PROJECT/script)],stdout=output,stderr=subprocess.STDOUT,check=True)
def main():
    try:
        phase('waiting_for_fold_predictions')
        while not (RUN/'cache_done.json').exists():time.sleep(15)
        phase('auditing_oof_lineage_and_training_inputs');run('audit_oof_cache.py','audit_oof.log')
        phase('training_residuals_and_sequence_disjoint_gates');jobs=[]
        for method,device in [('oof_dit','cuda:1'),('oof_regression','cuda:2'),('in_subject_dit','cuda:3')]:
            if (RUN/method/'done.json').exists():continue
            out=(RUN/f'train_{method}.log').open('w')
            process=subprocess.Popen([sys.executable,str(PROJECT/'train_oof_residual.py'),'--method',method,'--device',device],stdout=out,stderr=subprocess.STDOUT)
            jobs.append((method,process,out))
        for method,process,out in jobs:
            code=process.wait();out.close()
            if code:raise RuntimeError(f'{method} failed with exit code {code}')
        phase('checkpoint_calibration_frozen_final_evaluation')
        if not (RUN/'evaluation_done.json').exists():run('evaluate_oof_study.py','evaluate_oof.log')
        phase('building_delivery');run('report_oof_study.py','report_oof.log')
        save(RUN/'status.json',dict(stage='complete',complete=True,results=str(RUN/'final_results.json'),report=str(RUN/'report.html')))
    except Exception as error:
        save(RUN/'status.json',dict(stage='failed',complete=False,error=repr(error)));raise
if __name__=='__main__':main()

