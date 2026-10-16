"""Check trial identity, rates, finite channels, and exact native decimation."""
import json,pathlib,numpy as np,pandas as pd
root=pathlib.Path(__file__).resolve().parents[1];report=[]
def main():
    for duration in [15,30]:
     folder=root/f'data/release/{duration}s';parent=pd.read_csv(folder/'koopman_ready/koopman_125hz.csv')
     manifest=pd.read_csv(folder/'metadata/cv_fold_manifest.csv')
     assert len(manifest)==42 and manifest.case_id.nunique()==14
     assert manifest.groupby('case_id').run_index.nunique().eq(3).all()
     assert set(manifest.trial_uid)==set(parent.trial_uid)
     for rate in [125,25,5,1]:
      df=pd.read_csv(folder/f'koopman_ready/koopman_{rate}hz.csv')
      assert df.groupby('trial_uid').size().eq(duration*rate).all()
      assert not df.duplicated(['trial_uid','sample_index']).any()
      cols=['gap_m','follower_longitudinal_velocity_mps','leader_velocity_in_follower_longitudinal_mps']
      assert np.isfinite(df[cols].to_numpy()).all()
      for _,g in df.groupby('trial_uid'):assert np.allclose(np.diff(g.t_rel_s),1/rate,atol=1e-8)
      expected=pd.concat([g.iloc[::125//rate] for _,g in parent.groupby('trial_uid',sort=False)],ignore_index=True)
      # sample-rate/sample-index metadata legitimately changes with decimation.
      pd.testing.assert_frame_equal(df[cols+['trial_uid','t_rel_s']],expected[cols+['trial_uid','t_rel_s']],check_exact=True,check_dtype=False)
      report.append(dict(duration_s=duration,rate_hz=rate,trials=42,rows=len(df),passed=True))
    out=root/'outputs/checks';out.mkdir(parents=True,exist_ok=True);(out/'data_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__ == "__main__":
    main()
