import os, json, subprocess, shutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim

ROOT=Path(__file__).resolve().parents[1]
CFG=json.load(open(ROOT/"config/experiment_config.json"))
OUT=ROOT/CFG["output_dir"]; FIG=ROOT/CFG["figure_dir"]
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode!=0: raise RuntimeError(p.stderr[-4000:])
    return p.stdout

def ffprobe(path):
    s=run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,pix_fmt","-of","json",str(path)])
    return json.loads(s)["streams"][0]

def extract_frames(video, outdir, max_frames=300):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    for f in outdir.glob("*.png"): f.unlink()
    run(["ffmpeg","-y","-i",str(video),"-frames:v",str(max_frames),
         str(outdir/"%06d.png")])
    return sorted(outdir.glob("*.png"))

def encode_h264(frames_dir, out_mp4, qp, gop, refs=1, cabac=True, fps=30):
    # Frames are losslessly staged as an image sequence; H.264 compression is applied by libx264.
    cmd=["ffmpeg","-y","-framerate",str(fps),"-i",str(Path(frames_dir)/"%06d.png"),
         "-c:v","libx264","-preset","medium","-qp",str(qp),
         "-g",str(gop),"-keyint_min",str(gop),"-refs",str(refs),
         "-bf","0","-pix_fmt","yuv420p"]
    cmd += ["-coder","ac" if cabac else "vlc"]
    cmd += [str(out_mp4)]
    run(cmd)

def decode(video, outdir, max_frames=300):
    return extract_frames(video,outdir,max_frames)

def metrics(ref_dir, rec_dir):
    refs=sorted(Path(ref_dir).glob("*.png")); recs=sorted(Path(rec_dir).glob("*.png"))
    n=min(len(refs),len(recs))
    rows=[]
    for i in range(n):
        a=cv2.imread(str(refs[i])); b=cv2.imread(str(recs[i]))
        if a is None or b is None: continue
        if a.shape!=b.shape: b=cv2.resize(b,(a.shape[1],a.shape[0]))
        ag=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY); bg=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
        mse=float(np.mean((ag.astype(np.float32)-bg.astype(np.float32))**2))
        psnr=float("inf") if mse==0 else float(10*np.log10((255.0**2)/mse))
        s=float(ssim(ag,bg,data_range=255))
        corr=float(np.corrcoef(ag.ravel(),bg.ravel())[0,1])
        rows.append([i+1,mse,psnr,s,corr])
    return pd.DataFrame(rows,columns=["frame","MSE","PSNR_dB","SSIM","Correlation"])

def monotonicity(df):
    d=df["MSE"].diff().dropna()
    p=df["PSNR_dB"].diff().dropna()
    return {
      "MSE_non_decreasing_fraction":float((d>=0).mean()),
      "PSNR_non_increasing_fraction":float((p<=0).mean()),
      "MSE_end_minus_start":float(df.MSE.iloc[-1]-df.MSE.iloc[0]),
      "PSNR_end_minus_start":float(df.PSNR_dB.iloc[-1]-df.PSNR_dB.iloc[0])
    }

def plot_quality(df, name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11,7),dpi=800)
    plt.plot(df.frame,df.MSE,linewidth=2)
    plt.xlabel("Frame index",fontsize=18,fontweight="bold")
    plt.ylabel("MSE",fontsize=18,fontweight="bold")
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(False); plt.tight_layout()
    plt.savefig(FIG/f"{name}_MSE.png",dpi=800); plt.close()

    plt.figure(figsize=(11,7),dpi=800)
    plt.plot(df.frame,df.PSNR_dB,linewidth=2)
    plt.xlabel("Frame index",fontsize=18,fontweight="bold")
    plt.ylabel("PSNR (dB)",fontsize=18,fontweight="bold")
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.grid(False); plt.tight_layout()
    plt.savefig(FIG/f"{name}_PSNR.png",dpi=800); plt.close()

def frame_types(n,gop):
    types=[]
    for i in range(n):
        types.append("I" if i%gop==0 else "P")
    return types

def fixed_interval_ep(n,gop,interval):
    ep=[0]
    for i in range(1,n):
        if i%interval==0: ep.append(i)
    return sorted(set(ep))

def correlation_score(a,b):
    ag=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    bg=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    if np.std(ag)==0 or np.std(bg)==0: return 0.0
    return float(np.corrcoef(ag,bg)[0,1])

def aeps_positions(frames, gop, min_interval=3):
    # Practical frame-content implementation of the manuscript's correlation-driven AEPS logic.
    positions=[]
    for start in range(0,len(frames),gop):
        end=min(start+gop,len(frames))
        ref=frames[start]
        chosen=[start]
        last=start
        for i in range(start+min_interval,end):
            if i-last<min_interval: continue
            score=correlation_score(ref,frames[i])
            # local candidate selection; search only within remaining GOP
            if score>0.96:
                chosen.append(i); last=i; ref=frames[i]
        positions.extend(chosen)
    return sorted(set(positions))

def create_summary(df, config_name):
    s=df[["MSE","PSNR_dB","SSIM","Correlation"]].mean().to_dict()
    s["Configuration"]=config_name
    return s

def main():
    vids=list((ROOT/CFG["video_dir"]).glob("*"))
    vids=[v for v in vids if v.suffix.lower() in {".mp4",".yuv",".avi",".mov",".mkv"}]
    if not vids: raise FileNotFoundError("Put at least one real source video in data/.")
    video=vids[0]
    info=ffprobe(video)
    print("Input:",video,info)
    raw=ROOT/"results"/"frames_original"
    refs=extract_frames(video,raw,CFG["max_frames"])
    all_summaries=[]
    for qp in CFG["qps"]:
        enc=ROOT/"results"/f"h264_qp{qp}.mp4"
        rec=ROOT/"results"/f"decoded_qp{qp}"
        encode_h264(raw,enc,qp,CFG["gop_length"],CFG["reference_frames"],
                    CFG["entropy"].lower()=="cabac",CFG["fps"])
        decode(enc,rec,CFG["max_frames"])
        m=metrics(raw,rec)
        m.to_csv(OUT/f"frame_metrics_qp{qp}.csv",index=False)
        plot_quality(m,f"qp{qp}")
        q=m.copy(); q["QP"]=qp
        all_summaries.append(create_summary(m,f"H.264 QP={qp}"))
        m2=m.copy(); m2["GOP"]=((m2.frame-1)//CFG["gop_length"])+1
        m2["FrameType"]=frame_types(len(m2),CFG["gop_length"])
        m2.to_csv(OUT/f"frame_metrics_with_gop_qp{qp}.csv",index=False)
        print("QP",qp,monotonicity(m))
    pd.DataFrame(all_summaries).to_csv(OUT/"summary_metrics.csv",index=False)
    print("Completed. Results:",OUT)

if __name__=="__main__":
    main()
