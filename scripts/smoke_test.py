"""Numerical checks, independent Python/native agreement, and neural updates."""
import json,pathlib,sys,importlib
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'experiments'));sys.path.insert(0,str(root/'src'))
import revision_koopman as k,revision_physics as ph
from empirical_benchmark import load_trials
from protocol_revision_sensitivity import resample
from revision_data import crop
checks=k.unit_checks();assert max(x['acceleration_error'] for x in checks)<1e-12
tr=load_trials(30,125)[0][0];train=resample(tr,17,20,10);future=crop(tr,20,25)
assert len(train.t)==31 and len(future.t)==626
errors={}
for family,dictionary in [('CTH','quadratic'),('IDM','cth')]:
 coef,audit=ph.fit(train.t,train.z,train.u,family)
 prior=dict(kind='physical',family=family.lower(),coef=coef)
 model=k.fit(train.t,train.z,train.u,'expweak',dict(dictionary=dictionary,prior=family.lower(),prior_model=prior,T=1.,ridge=.1,kappa=1.))
 for name,m in [(family,prior),('PI-'+family,model)]:
  native=ph.predict_field(m,future.t,future.u,future.z[0,:2])
  reference=k.predict(m,future.t,future.u,future.z[0,:2],max_dt=.04)
  assert np.isfinite(native).all()
  error=float(np.max(np.abs(native-reference)));assert error<1e-8,(name,error)
  errors[name]=error
import revision_neural,revision_lstm
revision_lstm.OUT.mkdir(parents=True,exist_ok=True)
revision_neural.tests();revision_lstm.tests()
import run_pi_residual_recognition as recognition
recognition.synthetic_checks()
for name in ['unified_axes_neural','strengthen_encounter_staged']:
 importlib.import_module(name)
report=dict(weak_constant_acceleration_max_error=max(x['acceleration_error'] for x in checks),native_python_max_abs_errors=errors,neural_unit_checks=True,recognition_synthetic_checks=True)
(root/'outputs/checks').mkdir(parents=True,exist_ok=True);(root/'outputs/checks/smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
