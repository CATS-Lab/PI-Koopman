#!/usr/bin/env python3
"""Compact LSTM training on 30-s encounters."""
from pikoopman.models.lstm import *
OUT = ROOT/'outputs/neural/compact/lstm'

def run(dataset, budget, rate):
    tag = f'{dataset}_B{budget}_Hz{rate}_clean_0'; out = OUT/tag; out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.csv').exists():
        print('Already complete', tag, flush=True)
        return
    if dataset=='local':
        trials, path = load_trials(30, 125); meta = None; end = 20
    else:
        trials, meta = openacc(); path = ROOT/'data/external/openacc/episode_manifest.csv'; end = 20
    train = [crop(tr, end-budget, end, rate) for tr in trials]
    test = [crop(tr, end, end+5) for tr in trials]
    n = len(trials); m = n*len(SEEDS)
    raw = np.array([tr.z for seed in SEEDS for tr in train], dtype=np.float32)
    leader = np.array([tr.u for seed in SEEDS for tr in train], dtype=np.float32)
    mu = raw.mean(axis=1, keepdims=True)
    scale = np.maximum(raw.std(axis=1, keepdims=True), np.array([1., .5, .5], np.float32))
    normalized = (raw-mu)/scale
    horizon = int(round(.4*rate)); stride = horizon
    starts = np.arange(0, raw.shape[1]-horizon, stride)
    targets = torch.tensor(np.stack([normalized[:, j:j+horizon+1] for j in starts], axis=1))
    inputs = torch.tensor(np.stack([leader[:, j:j+horizon+1] for j in starts], axis=1))
    mt, st = torch.tensor(mu), torch.tensor(scale)
    dt = float(np.median(np.diff(train[0].t)))
    config = dict(dataset=dataset, budget_s=budget, rate_hz=rate, scenario='clean', corruption_seed=0,
                  neural_initializations=SEEDS, checkpoints=CHECKPOINTS, families=['LSTM'],
                  parameters=393, recurrent_units=8, observed_step_s=dt,
                  training_horizon_samples=horizon, training_horizon_s=horizon*dt,
                  sequence_stride_samples=stride, n_sequences_per_trial=len(starts),
                  train_samples=raw.shape[1], train_start_s=end-budget, train_end_s=end,
                  no_hidden_high_rate_calibration_targets=True, data_file=str(path),
                  data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  protocol_file=str(OUT/'protocol_frozen.json'),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(config, indent=2))
    net = initialize(n)
    opt = torch.optim.Adam(net.parameters(), lr=.003, weight_decay=1e-5)
    rows, predictions, loss_rows, audits = [], [], [], []
    training_failed = torch.zeros(m, dtype=torch.bool)
    started = time.perf_counter()
    for epoch in range(1, max(CHECKPOINTS)+1):
        opt.zero_grad(); state = targets[:, :, 0]; memory = None
        loss_per = torch.zeros(m)
        for k in range(horizon):
            acc, memory = net.acceleration(state, memory)
            state = advance(state, acc, inputs[:, :, k], inputs[:, :, k+1], dt, mt, st)
            loss_per += .9**k*(state-targets[:, :, k+1]).square().mean((1, 2))
        training_failed |= ~torch.isfinite(loss_per)
        loss = torch.where(training_failed, torch.zeros_like(loss_per), loss_per).sum()/horizon
        loss.backward()
        norms = torch.zeros(m)
        for p in net.parameters():
            p.grad.nan_to_num_(nan=0., posinf=0., neginf=0.)
            p.grad[training_failed] = 0.
            norms += p.grad.reshape(m, -1).square().sum(1)
        factor = torch.clamp(10/(norms.sqrt()+1e-9), max=1.)
        for p in net.parameters():
            p.grad *= factor.reshape(m, *([1]*(p.ndim-1)))
        opt.step()
        if epoch%50==0:
            print(tag, 'epoch', epoch, 'loss', float(loss.detach()/m), 'seconds', round(time.perf_counter()-started, 1), flush=True)
        if epoch not in CHECKPOINTS:
            continue
        fit_seconds = time.perf_counter()-started
        archive = dict(state_dict=net.state_dict(), mu=mu, scale=scale, uids=[tr.uid for tr in trials],
                       neural_seeds=SEEDS, family='LSTM', epoch=epoch, config=config,
                       training_failed=training_failed.numpy())
        torch.save(archive, out/f'LSTM_epoch{epoch}.pt')
        pred, audit = rollout(net, train, test, mu, scale, training_failed.numpy())
        audits.append(dict(family='LSTM', epoch=epoch, **audit))
        # Replay the saved weights before metrics are used for checkpoint choice.
        saved = torch.load(out/f'LSTM_epoch{epoch}.pt', weights_only=False)
        replay_net = initialize(n); replay_net.load_state_dict(saved['state_dict'])
        replay, _ = rollout(replay_net, train, test, saved['mu'], saved['scale'], saved['training_failed'])
        assert np.array_equal(pred, replay, equal_nan=True)
        cfg = json.dumps(dict(epoch=epoch, hidden_units=8, parameters=393,
                              recurrent_interval_s=1/rate, training_horizon_s=.4,
                              projection='d=u-v,gap_trapezoid', acceleration_hold='one observation interval'), sort_keys=True)
        for si, seed in enumerate(SEEDS):
            for i, tr in enumerate(trials):
                ix = si*n+i
                base = dict(dataset=dataset, trial_uid=tr.uid, case_id=tr.case, mode=tr.mode,
                            group=str(tr.run) if dataset=='local' else tr.case, budget_s=budget, rate_hz=rate,
                            scenario='clean', seed=0, neural_seed=seed, model='LSTM', config=cfg, parameters=393,
                            fit_seconds=fit_seconds/m, batch_fit_seconds=fit_seconds,
                            optimizer_success=not bool(training_failed[ix]), training_failed=int(training_failed[ix]))
                if meta is not None:
                    base.update(campaign=str(meta.iloc[i].campaign), vehicle=str(meta.iloc[i].vehicle))
                rows.append(dict(**base, **metric(pred[ix], test[i])))
                predictions.append(pred[ix])
                loss_rows.append(dict(trial_uid=tr.uid, model='LSTM', neural_seed=seed, epoch=epoch,
                                      train_loss=float(loss_per[ix].detach()/horizon), batch_fit_seconds=fit_seconds))
        print(tag, 'saved', epoch, 'failures', audit['failures'], 'replay exact', flush=True)
    df = pd.DataFrame(rows); df.to_csv(out/'candidate_metrics.csv', index=False)
    np.savez_compressed(out/'candidate_predictions.npz', predictions=np.array(predictions))
    pd.DataFrame(loss_rows).to_csv(out/'training_loss.csv', index=False)
    selected, choices = select(df)
    ss = df.loc[selected].copy(); ss.to_csv(out/'selected_metrics.csv', index=False)
    agg = ss.groupby(['dataset', 'trial_uid', 'case_id', 'mode', 'group', 'budget_s', 'rate_hz', 'scenario', 'seed', 'model'], as_index=False).agg(
        speed_rmse_mps=('speed_rmse_mps', 'mean'), gap_rmse_m=('gap_rmse_m', 'mean'), score=('score', 'mean'),
        failed_any_seed=('failed', 'max'), n_initializations=('neural_seed', 'count'), fit_seconds_amortized=('fit_seconds', 'mean'))
    agg.to_csv(out/'trial_metrics.csv', index=False)
    agg.groupby('model', sort=False).agg(n_trials=('trial_uid', 'size'), speed_rmse_mps=('speed_rmse_mps', 'mean'),
        gap_rmse_m=('gap_rmse_m', 'mean'), score=('score', 'mean'), failures_any_seed=('failed_any_seed', 'sum'),
        n_initializations=('n_initializations', 'min')).reset_index().to_csv(out/'summary.csv', index=False)
    (out/'selection.json').write_text(json.dumps(choices, indent=2))
    (out/'kinematic_projection_audit.json').write_text(json.dumps(audits, indent=2))
    np.savez_compressed(out/'selected_predictions.npz', predictions=np.array(predictions)[selected],
                        trial_uid=ss.trial_uid.to_numpy(), model=ss.model.to_numpy(), neural_seed=ss.neural_seed.to_numpy())
    config.update(total_seconds=time.perf_counter()-started, torch=torch.__version__, n_selected_models=len(ss),
                  all_selection_groups_disjoint=True, all_checkpoint_weight_replays_exact=True,
                  candidate_failures=int(df.failed.sum()), selected_failures=int(ss.failed.sum()))
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print('DONE', tag, config['total_seconds'], flush=True)
