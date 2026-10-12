"""Validate named columns, compute TDI before seasonal filtering, and audit supplied features."""
import json
import numpy as np
import pandas as pd
from project_config import ROOT, CFG, PROCESSED_DIR, STAGE_OUTPUT
from tdi import add_daily_tdi_features

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
audit = {}
references = {'XB10':'xb10-TDI.xlsx', 'LHJ2':'LHJ-TDI6.xlsx', 'A114':'A114-TDI.xlsx'}
for sample, filename in {'XB10':'XB10.xlsx', 'LHJ2':'LHJ.xlsx', 'A114':'A114.xlsx'}.items():
    df = pd.read_excel(ROOT/'data/raw'/filename)
    df = df[['date','DGR','DGRV','label']].copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df = df.sort_values('date').reset_index(drop=True)
    if df['date'].duplicated().any() or df.isna().any().any():
        raise ValueError(f'{sample}: missing values or duplicate dates')
    if not df['label'].isin([0,1]).all():
        raise ValueError(f'{sample}: labels must be binary')
    if not np.isfinite(df[['DGR','DGRV']].to_numpy()).all():
        raise ValueError(f'{sample}: nonfinite features')
    if not df['date'].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError(f'{sample}: raw TDI input must be a continuous daily sequence')
    if len(df) < CFG['tdi_window']:
        raise ValueError('The original TDI implementation requires at least one full window')
    out = add_daily_tdi_features(df, window_size=CFG['tdi_window'])
    if sample == 'LHJ2':
        out = out.loc[out['date'].dt.month.between(6,11)].copy()
    ordered = ['date','DGR','DGRV','TDI_A','TDI_V','TDI_F','TDI_E','edge_flag','label']
    out = out[ordered].reset_index(drop=True)
    out.to_excel(PROCESSED_DIR/(sample+'.xlsx'), index=False)
    out.to_csv(PROCESSED_DIR/(sample+'.csv'), index=False)
    ref = pd.read_excel(ROOT/'data/reference_features'/references[sample])
    ref['date'] = pd.to_datetime(ref['date'])
    matched = out.merge(ref, on='date', suffixes=('_new','_ref'), validate='one_to_one')
    differences = {col:float(np.max(np.abs(matched[col+'_new']-matched[col+'_ref']))) for col in ordered if col not in ['date','edge_flag']}
    audit[sample] = {'raw_rows':len(df), 'processed_rows':len(out),
                     'label_counts':out['label'].value_counts().sort_index().to_dict(),
                     'reference_rows':len(ref), 'matched_dates':len(matched),
                     'max_absolute_difference_from_supplied_features':differences}
    print(sample, audit[sample])
(STAGE_OUTPUT/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
