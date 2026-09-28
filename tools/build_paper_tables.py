#!/usr/bin/env python3
"""Write values transcribed from the supplied JSA_main.pdf, not newly measured data.

DOI: 10.1016/j.sysarc.2026.103836. Preserve published rounding and discrepancies.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/paper'


def table(name, header, rows, location):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header.split(',') + ['provenance', 'paper_location'])
        writer.writerows([list(row) + ['transcribed_from_supplied_paper_not_rerun', location] for row in rows])


def main():
    table('table1_models.csv', 'model,blocks,embedding_dim,mlp_dim,heads,parameters_millions_reported,flops_billions_reported,fp32_top1_pct,fx16_top1_pct', [
        ['ViT-Tiny',12,192,768,3,9.7,1.1,66.46,66.15], ['DeiT-Tiny',12,192,768,3,5.0,1.3,62.31,61.83],
        ['ViT-Small',12,384,1536,6,30.1,4.3,78.22,78.01], ['DeiT-Small',12,384,1536,6,22.0,4.6,74.11,73.87],
        ['ViT-Base',12,768,3072,12,86.0,16.9,83.54,83.32], ['ViT-Large',24,1024,4096,16,307,61.6,84.82,84.64]], 'Table 1; page 7')
    configs = ['0/0/0/0','10/10/10/10','15/10/10/10','10/15/15/15','15/15/15/15','15/20/20/20','20/15/15/15','20/20/20/20','20/25/25/25']
    values = {
        'ViT-Base': [(83.32,17.571,0),(83.16,13.811,21.39),(83.14,13.010,25.91),(82.99,12.930,26.38),(83.02,12.160,30.80),(82.42,11.440,34.90),(82.34,11.470,34.73),(82.02,10.770,38.73),(81.66,10.040,42.85)],
        'ViT-Small': [(78.01,4.603,0),(77.44,3.59,21.86),(76.97,3.38,26.44),(77.29,3.36,26.88),(76.82,3.16,31.34),(75.98,2.97,35.44),(76.23,2.97,35.31),(75.38,2.79,39.30),(74.27,2.61,43.18)],
        'ViT-Tiny': [(66.15,1.255,0),(64.81,.97,22.68),(63.74,.91,27.31),(64.52,.90,27.76),(63.27,.86,31.35),(62.56,.81,36.36),(61.94,.81,38.85),(61.29,.76,40.27),(60.08,.71,44.08)]}
    rows = []
    for model, entries in values.items():
        for config, (acc,gops,savings) in zip(configs, entries):
            rows.append([model,config,acc,round(acc-entries[0][0],2),gops,savings])
    table('table2_pruning.csv', 'model,drop_percent_at_minus1_3_6_9,top1_pct,accuracy_change_pp,gops_reported,savings_pct_reported', rows, 'Table 2; page 7')
    table('table3_pruning_hardware.csv', 'module,precision,frequency_mhz,bram,dsp,ff,lut,latency_ms', [
        ['EDPG','FX16',250,5,256,37194,105142,.637], ['ATS (ViT-Small)','FX16',250,136,0,97665,38793,.254]], 'Table 3; page 8')
    rows=[]
    for stage, accs in [('EDPG -1',[74.53,74.78,74.82]),('ATS 0',[72.72,72.39,72.23]),('ATS 1',[78.71,78.83,78.79]),('ATS 2',[75.87,76.11,76.01]),('ATS 3',[79.32,79.55,79.89])]:
        for model,base,acc in zip(['Swin-Tiny','Swin-Small','Swin-Base'],[80.31,80.40,80.48],accs):
            rows.append([model,stage,50,acc,round(acc-base,2)])
    table('table4a_swin_independent.csv','model,pruning_stage,drop_pct,top1_pct,accuracy_change_pp',rows,'Table 4a; page 9')
    rows=[]
    for rates, accs in [('0/0',[80.31,80.40,80.48]),('10/10',[79.66,79.52,79.40]),('15/15',[79.12,79.13,79.18]),('15/20',[78.96,78.64,78.63]),('20/15',[78.45,78.26,78.34]),('20/20',[78.59,78.35,78.55]),('25/25',[77.52,77.76,77.79])]:
        for model,base,acc in zip(['Swin-Tiny','Swin-Small','Swin-Base'],[80.31,80.40,80.48],accs):
            rows.append([model,rates,acc,round(acc-base,2)])
    table('table4b_swin_combined.csv','model,drop_percent_at_minus1_1,top1_pct,accuracy_change_pp',rows,'Table 4b; page 9')
    table('table5_hardware.csv','method,model,platform,precision,frequency_mhz,pruning_pct,bram,dsp,ff,lut,power_w,fps,fps_per_w,accuracy_change_pp',[
        ['HeatViT','DeiT-Small','ZCU102','FX8',150,'10/16/39',338.5,1955,100400,145000,10.697,57,5.32,''],
        ['AutoViT no quantization','ViT-Small','ZCU102','W32A32',150,'','',1745,'',130000,8.38,38.9,4.64,0],
        ['AutoViT quantization','ViT-Small','ZCU102','W4A4+W3A4',150,'','',1549,'',193000,10.34,155.8,15.06,-1.91],
        ['MoE-Sched','ViT-Small','U280','INT8','','',652,3379,'','',39.24,119.3,3.04,-.28],
        *[['GateAttn-ViT','ViT-Small','ZCU104','FX16',250,rate,505,1248,86122,58640,3.88,fps,eff,loss] for rate,fps,eff,loss in [('0/0/0/0',14.20,3.65,0),('10/10/10/10',15.73,4.05,-.57),('20/20/20/20',19.63,5.06,-2.63),('20/25/25/25',20.91,5.39,-3.74)]],
        ['GateAttn-ViT','ViT-Small','ZCU104','FX16',150,'17/17/17/17',457,848,20297,52489,3.76,12.08,3.21,-.37]],'Table 5; page 10')
    table('table6_comparison.csv','method,model,pruning_pct,blocks,precision,top1_pct,accuracy_change_pp,gops_reported,savings_pct_reported',[
        ['ViT-to-Go','ViT-Base',30,'4/6/8','FX16',78.21,-1.84,11.14,36.59],
        ['ViT-to-Go','ViT-Base',20,'4/6/8','FX16',79.60,-.45,12.86,26.83],
        ['HeatViT-B1','DeiT-Base','','','FX8',81.05,-.72,10.49,40.4],
        ['GateAttn-ViT','ViT-Base',15,'-1/3/6/9','FX16',83.02,-.30,12.160,30.80],
        ['GateAttn-ViT','ViT-Base',20,'-1/3/6/9','FX16',82.02,-1.30,10.770,38.73]],'Table 6; pages 10–11')
    rows=[]
    for model,cifar,obj,saving in [
        ('ViT-Tiny',[76.48,75.52,74.53,72.07,68.42,63.61],[60.61,60.53,60.38,59.35,58.55,57.04],[0,12.40,22.13,32.26,41.81,50.74]),
        ('ViT-Small',[84.93,84.33,82.75,80.66,77.65,71.75],[63.47,63.24,62.86,62.33,61.74,60.69],[0,10.63,21.31,31.83,41.47,50.5])]:
        for i,rate in enumerate(range(0,51,10)):
            rows.append([model,rate,cifar[i],obj[i],saving[i]])
    table('table7_datasets.csv','model,drop_pct,cifar100_top1_pct,objectnet_top1_pct,savings_pct_reported',rows,'Table 7; page 11')
    rows=[]
    for model,crop,random,entropy in [
        ('ViT-Tiny',[66.15,65.02,59.30,56.16,51.90,46.29],[66.15,64.50,61.97,59.61,54.64,47.11],[66.15,65.27,63.12,60.17,56.04,49.17]),
        ('ViT-Small',[78.11,77.44,75.89,75.05,72.10,66.87],[78.11,76.99,75.79,73.68,70.68,65.91],[78.11,77.66,76.61,74.95,72.42,68.14])]:
        for i,rate in enumerate(range(0,51,10)):
            rows.append([model,rate,crop[i],random[i],entropy[i]])
    table('table8_baselines.csv','model,drop_pct,center_crop_top1_pct,random_drop_top1_pct,edpg_top1_pct',rows,'Table 8; page 11')
    table('figure9_latency.csv','uniform_pruning_pct,tokenization_ms,msa_ms,mlp_ms,layernorm_ms,residual_ms,vit_io_ms,total_ms_reported',[
        [0,.683,22.008,45.108,1.368,.696,.572,70.435],
        [10,.683,20.412,40.020,1.176,.696,.572,63.559],
        [20,.683,15.816,32.196,.984,.696,.572,50.947]],'Figure 9 and Section 4.7; pages 9–10')
    print('Wrote 9 table CSVs and Figure 9 latency data.')


if __name__ == '__main__':
    main()
