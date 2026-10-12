"""Verify reference checkpoints against archived daily probabilities without retraining."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from project_config import CFG, FEATURES, MODEL_DIR, PROCESSED_DIR, ROOT, STAGE_OUTPUT
from model_core import BiLSTMAttention, build_sliding_window

torch.set_num_threads(2)
summary = {}
for sample in ['XB10','LHJ2']:
    df = pd.read_excel(PROCESSED_DIR/(sample+'.xlsx')).sort_values('date').reset_index(drop=True)
    heldout = df.iloc[int(len(df)*(1-CFG['heldout_fraction'])):].copy()
    slug = sample.lower()
    model_path = MODEL_DIR/(slug+'_best.pth')
    scaler_path = MODEL_DIR/('scaler_'+slug+'.pkl')
    scaler = joblib.load(scaler_path)
    model = BiLSTMAttention(len(FEATURES), CFG['hidden_size'])
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    x, y = build_sliding_window(scaler.transform(heldout[FEATURES]), heldout['label'].to_numpy(), heldout['date'], CFG['left_window'], CFG['right_window'])
    with torch.no_grad():
        probability = torch.softmax(model(torch.from_numpy(x)),dim=1)[:,1].numpy()
    predicted = (probability >= CFG['classification_threshold']).astype(int)
    report = classification_report(y,predicted,labels=[0,1],target_names=['Non-TC','TC'],output_dict=True,zero_division=0)
    daily = pd.DataFrame({'date':heldout['date'].to_numpy(),'true_label':y,'tc_probability':probability,'predicted_label':predicted})
    daily.to_csv(STAGE_OUTPUT/(slug+'_daily_probability.csv'),index=False)
    pd.DataFrame(report).T.to_csv(STAGE_OUTPUT/(slug+'_classification_report.csv'))
    ref = pd.read_csv(ROOT/'reference_results'/(slug+'_test_daily_probability.csv'),parse_dates=['date'])
    matched = daily.merge(ref,on='date',suffixes=('_new','_ref'),validate='one_to_one')
    if len(matched) != len(daily) or len(matched) != len(ref):
        raise ValueError('Reference dates differ from evaluation dates')
    difference = float(np.max(np.abs(matched.tc_probability_new-matched.tc_probability_ref)))
    agreement = bool(np.array_equal(matched.predicted_label_new,matched.predicted_label_ref))
    summary[sample] = {'heldout_rows':len(y),'tc_f1':report['TC']['f1-score'], 'accuracy':report['accuracy'],
                       'max_probability_difference_from_archive':difference,
                       'all_predicted_labels_match_archive':agreement,
                       'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                       'scaler_sha256':hashlib.sha256(scaler_path.read_bytes()).hexdigest()}
    print(sample,summary[sample])
(STAGE_OUTPUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if not all(r['all_predicted_labels_match_archive'] and r['max_probability_difference_from_archive'] < 1e-5 for r in summary.values()):
    raise SystemExit('Reference verification differs; see verification.json. Do not claim exact reproduction.')
