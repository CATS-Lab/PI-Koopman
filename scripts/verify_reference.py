"""Compare completed main-text synthetic and recognition outputs to frozen references."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
checks=[('synthetic/field_error.csv',['system','rate_hz','speed_noise_sd_mps','seed','method'],['field_rmse_mps2']),
('pi_recognition_2026-10-15/rows.csv',['trial_uid','model'],['residual_variance','residual_rmse']),
('pi_recognition_2026-10-15/oof.csv',['trial_uid','model','cohort','score_name'],['threshold','score_value','predicted_acc'])]
for rel,keys,values in checks:
 a=pd.read_csv(root/'results'/rel).sort_values(keys).reset_index(drop=True)
 b=pd.read_csv(root/'reference'/rel).sort_values(keys).reset_index(drop=True)
 pd.testing.assert_frame_equal(a[keys],b[keys],check_dtype=False)
 pd.testing.assert_frame_equal(a[values],b[values],atol=1e-9,rtol=1e-8,check_dtype=False)
 print('PASS',rel,len(a))
