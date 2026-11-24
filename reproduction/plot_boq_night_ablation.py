"""Plot completed night-selected BoQ results; no training or model selection."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path('runs/boq_night_selected_multi_night_v1')
FAMILIES=[['tokyo105_night'],['svox_test_night'],
 ['robotcar_seasons_public_night','robotcar_seasons_public_night_rain'],
 ['gardenspoint_night_day_right','gardenspoint_night_day_left'],['msls_test_night_day']]
PRIMARY=[p for family in FAMILIES for p in family]
EXTRA=['darkdriving_paired_diagnostic','sped_mixed_illumination_diagnostic']
ZH=dict(saturation='降饱和',white_balance='白平衡',defocus='失焦',resolution='降分辨率',
 motion_blur='运动模糊',noise='噪声',jpeg='JPEG',gain_gamma='增益/Gamma',
 local_exposure='局部曝光',vignette='暗角',bloom='泛光')


def main():
    source=ROOT/'summary.json';s=json.loads(source.read_text())
    assert s['status']=='completed' and len(s['models'])==14
    def r(model,p,k='R@1'):return s['results'][p][model]['metrics_percent'][k]
    def macro(model,k='R@1'):
        return float(np.mean([np.mean([r(model,p,k) for p in family]) for family in FAMILIES]))
    def delta(model,reference):
        return [macro(model)-macro(reference)]+[r(model,p)-r(reference,p) for p in PRIMARY+EXTRA]
    def paired(model,reference,p):
        a=np.load(ROOT/p/f'{model}_hits.npy')[:,0]
        b=np.load(ROOT/p/f'{reference}_hits.npy')[:,0]
        assert len(a)==s['protocols'][p]['queries']['count']
        assert abs(a.mean()*100-r(model,p))<1e-8
        return dict(queries=len(a),gained=int((a & ~b).sum()),lost=int((b & ~a).sum()),
                    net=int(a.sum()-b.sum()))
    out=dict(summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
       families=FAMILIES,primary=PRIMARY,supplementary=EXTRA,
       policy='Night dev selects each checkpoint; test comparisons are descriptive, not new model selection',
       models={m:dict(epoch=v['epoch'],night_dev_R1=v.get('night_dev_R1'),
          family_macro={k:macro(m,k) for k in ['R@1','R@5','R@10']},
          delta_vs_initial=delta(m,'baseline'),delta_vs_full_C=delta(m,'C_no_distill'))
          for m,v in s['models'].items()},
       full_C_vs_no_degradation={p:paired('C_no_distill','no_degradation',p) for p in PRIMARY+EXTRA},
       full_C_vs_initial={p:paired('C_no_distill','baseline',p) for p in PRIMARY+EXTRA})
    (ROOT/'ablation_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    plt.rcParams.update({'font.family':FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name(),
        'axes.unicode_minus':False,'font.size':12,'pdf.fonttype':42})
    omissions=sorted([m for m in s['models'] if m.startswith('no_') and m!='no_degradation'],
                      key=lambda m:-macro(m))
    rows=[('C_no_distill','完整 C'),('no_degradation','不加任何退化')]
    bottom=[(m,'去除'+ZH[m.removeprefix('no_')]) for m in omissions]+[('no_degradation','不加任何退化')]
    fig,axes=plt.subplots(2,1,figsize=(20,14),gridspec_kw={'height_ratios':[2,12]})
    fig.subplots_adjust(left=.13,right=.91,top=.91,bottom=.13,hspace=.24)
    columns=['五家族\n平均','Tokyo105','SVOX','RobotCar\nNight','RobotCar\nNight-rain',
             'Gardens\n同侧','Gardens\n跨侧','MSLS\ntest','DarkDriving\n配对诊断','SPED\n混合光照']
    for ax,entries,ref,title in zip(axes,[rows,bottom],['baseline','C_no_distill'],
        ['整体收益：相对初始 DINOv2-BoQ','逐项消融：相对完整 C（蓝色表示去掉后更好）']):
        values=np.array([delta(m,ref) for m,_ in entries])
        im=ax.imshow(values,cmap='RdBu',vmin=-4.5,vmax=4.5,aspect='auto')
        ax.set_xticks(range(len(columns)),columns)
        ax.set_yticks(range(len(entries)),[label for _,label in entries])
        ax.set_title(title,loc='left',pad=13,fontsize=16)
        ax.tick_params(length=0,pad=9)
        ax.axvline(.5,color='#444',linewidth=1)
        ax.axvline(7.5,color='#444',linewidth=1.5,linestyle='--')
        if len(entries)>2:ax.axhline(len(entries)-1.5,color='#444',linewidth=1.5)
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,'0.00' if abs(v)<.005 else f'{v:+.2f}',ha='center',va='center',
                    color='white' if abs(v)>3.2 else '#152536',fontsize=12)
    fig.suptitle('DINOv2-BoQ 夜间退化消融｜全部无蒸馏',fontsize=23,y=.975)
    fig.text(.13,.94,'各组同为 10 轮；按 SVOX 夜间开发集选模；测试集只报告结果',fontsize=13)
    cax=fig.add_axes([.935,.21,.016,.58]);cb=fig.colorbar(im,cax=cax)
    cb.set_label('R@1 变化（百分点）',labelpad=15)
    fig.text(.13,.045,'五家族平均：Tokyo、SVOX、RobotCar、Gardens、MSLS 等权；RobotCar/Gardens 内部先取平均。\n'
             '虚线右侧为补充诊断，不计入五家族平均；逐项消融按平均变化排序。单次训练结果不代表统计显著性。',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'ablation_transfer_r1.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    for m,v in out['models'].items():
        print(m,'epoch',v['epoch'],'macro',round(v['family_macro']['R@1'],4),
              'vsC',round(v['delta_vs_full_C'][0],4))
    print('paired full C vs no degradation',out['full_C_vs_no_degradation'])


if __name__=='__main__':main()
