
import io, os, json, random, zipfile, tempfile, shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

CLASSES = ["Coca Cola", "Sprite", "Pepsi", "Mountain Dew", "7UP", "Fanta"]
EXTS = {".jpg", ".jpeg", ".png"}
METHODS = {"lighting":"Lighting", "camera_angle":"Camera angle", "perspective":"Perspective", "combined":"Combined"}
st.set_page_config(page_title="ShelfAugment AI", page_icon="🛒", layout="wide")

CSS = (
'<style>'
':root{color-scheme:light;--navy:#27374D;--blue:#526D82;--steel:#9DB2BF;--pale:#DDE6ED;--paper:#F7FAFC;--line:#C4D1DB;--ink:#27374D;--muted:#526D82}'
'.stApp{background:var(--pale);color:var(--ink)}'
'[data-testid="stHeader"]{background:rgba(221,230,237,.98);height:3.6rem;z-index:1000;border-bottom:1px solid #C4D1DB}'
'[data-testid="stToolbar"]{top:.35rem;right:.75rem;z-index:1100}'
'[data-testid="stToolbar"] button{background:#DDE6ED!important;border:1px solid #9DB2BF!important;border-radius:9px!important;color:#27374D!important;box-shadow:0 1px 3px rgba(39,55,77,.10)!important;margin-left:5px!important}'
'[data-testid="stToolbar"] button:hover{background:#9DB2BF!important;border-color:#526D82!important;color:#27374D!important}'
'[data-testid="stToolbar"] button svg{color:#526D82!important;fill:none!important;stroke:#526D82!important}'
'[data-testid="stToolbar"] button:hover svg{color:#27374D!important;stroke:#27374D!important}'
'[data-testid="stToolbar"] [data-testid="stToolbarActionButton"]{background:#DDE6ED!important;border-radius:9px!important}'
'[data-testid="stAppViewContainer"]>.main{padding-top:.2rem}'
'[data-testid="stSidebar"]{background:#DDE6ED;border-right:1px solid #9DB2BF}'
'[data-testid="stSidebar"] *{color:#27374D!important}'
'[data-testid="stSidebar"] [data-testid="stRadio"] label{font-size:14px!important;font-weight:500!important}'
'.block-container{padding-top:4.1rem;padding-bottom:3rem;max-width:1480px}'
'h1,h2,h3,h4{color:#27374D!important;letter-spacing:-.02em}'
'p,li,span,small{color:inherit}'
'.brand{display:flex;align-items:center;gap:14px;padding:18px 22px;background:#27374D;border-radius:17px;color:#F7FAFC;margin-bottom:20px}'
'.brandmark{width:46px;height:46px;display:grid;place-items:center;border-radius:13px;background:#9DB2BF;color:#27374D;font-size:25px}'
'.brandtitle{font-size:25px;font-weight:750;color:#F7FAFC;line-height:1.15}.brandsub{font-size:11px;color:#DDE6ED;letter-spacing:.12em;text-transform:uppercase;margin-top:5px}'
'.kicker{font-size:11px;color:#526D82;font-weight:750;letter-spacing:.13em;text-transform:uppercase}'
'.metriccard{background:#F7FAFC;border:1px solid #C4D1DB;border-radius:14px;padding:17px;min-height:108px;color:#27374D}'
'.metriclabel{font-size:13px;color:#526D82;font-weight:700;line-height:1.35;white-space:normal;overflow-wrap:anywhere}.metricvalue{font-size:29px;color:#27374D;font-weight:750;margin-top:7px}'
'.metricnote{font-size:11px;color:#526D82;margin-top:6px}.panel{background:#F7FAFC;border:1px solid #C4D1DB;border-radius:15px;padding:17px 19px;margin:8px 0 15px;color:#27374D}'
'.softnote{background:#F7FAFC;border:1px solid #C4D1DB;border-left:4px solid #526D82;border-radius:9px;padding:12px 15px;color:#27374D;font-size:13px}'
'div.stButton>button,div.stDownloadButton>button{background:#526D82!important;color:#FFFFFF!important;border:1px solid #526D82!important;border-radius:10px;font-weight:650;min-height:42px}'
'div.stButton>button:hover,div.stDownloadButton>button:hover{background:#27374D!important;color:#FFFFFF!important;border-color:#27374D!important}'
'div.stButton>button:disabled{background:#E7EDF2!important;color:#687D8D!important;border-color:#C4D1DB!important}'
'div.stButton>button[kind="secondary"]{background:#F7FAFC!important;color:#27374D!important;border-color:#9DB2BF!important}'
'[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,label{color:#27374D!important;font-weight:600!important}'
'[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:#526D82!important}'
'[data-testid="stFileUploader"]{background:#F7FAFC!important;border:1px dashed #9DB2BF!important;border-radius:12px;padding:9px;color:#27374D!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{background:#F7FAFC!important;color:#27374D!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] *{color:#27374D!important;-webkit-text-fill-color:#27374D!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button{background:#E7EEF3!important;color:#27374D!important;border:1px solid #9DB2BF!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]{background:#171717!important;background-color:#171717!important;color:#FFFFFF!important;border:1px solid #171717!important;border-radius:10px!important;padding:8px 10px!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *{color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;opacity:1!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] svg{color:#FFFFFF!important;fill:none!important;stroke:#FFFFFF!important}'
'[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button{background:#526D82!important;background-color:#526D82!important;color:#FFFFFF!important;border:1px solid #9DB2BF!important;border-radius:50%!important}'
'[data-testid="stFileUploader"] small{color:#27374D!important;-webkit-text-fill-color:#27374D!important;font-weight:600!important}'
'[data-testid="stSelectbox"] div[data-baseweb="select"]>div,[data-testid="stMultiSelect"] div[data-baseweb="select"]>div{background:#F7FAFC!important;color:#27374D!important;border-color:#9DB2BF!important}'
'[data-testid="stSelectbox"] *,[data-testid="stMultiSelect"] *{color:#27374D!important;font-weight:600}'
'[role="listbox"],[role="option"]{background:#F7FAFC!important;color:#27374D!important}'
'[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea{background:#F7FAFC!important;color:#27374D!important;border-color:#9DB2BF!important}'
'[data-testid="stSlider"] [data-testid="stTickBar"]{background:#9DB2BF}'
'[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#27374D!important}'
'[data-testid="stMetric"]{background:#F7FAFC;border:1px solid #C4D1DB;border-radius:12px;padding:12px}'
'[data-testid="stMetric"] label,[data-testid="stMetric"] [data-testid="stMetricLabel"]{color:#526D82!important}'
'[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#27374D!important}'
'[data-testid="stDataFrame"],[data-testid="stTable"]{background:#F7FAFC;border:1px solid #C4D1DB;border-radius:11px;overflow:hidden}'
'[data-testid="stDataFrame"] *{color:#27374D}'
'[data-testid="stExpander"]{background:#F7FAFC;border:1px solid #C4D1DB;border-radius:11px}'
'[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#27374D!important;font-weight:600}'
'[data-testid="stAlert"]{color:#27374D}'
'hr{border-color:#9DB2BF!important}footer,#MainMenu{visibility:hidden}'
'@media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}.brand{padding:14px;flex-wrap:wrap}.brandtitle{font-size:21px}.brandsub{font-size:9px}.metriccard{min-height:96px;padding:12px}.metricvalue{font-size:25px}}'
'</style>'
)
st.markdown(CSS, unsafe_allow_html=True)

def init():
    defaults={"history":[],"images_processed":0,"augmented_samples":0,"batch_runs":0,
      "original_boxes":0,"generated_boxes":0,"method_counts":Counter(),"single_result":None,
      "batch_df":None,"batch_zip":None,"batch_report":None,"qa_df":None,"qa_details":None,"tf_zip":None,"tf_report":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
init()

def add_event(row):
    st.session_state.history.insert(0,row)
    st.session_state.history=st.session_state.history[:30]

def card(label,value,note):
    st.markdown(f'<div class="metriccard"><div class="metriclabel">{label}</div><div class="metricvalue">{value}</div><div class="metricnote">{note}</div></div>',unsafe_allow_html=True)

def decode(upload):
    a=np.frombuffer(upload.getvalue(),np.uint8); im=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if im is None: raise ValueError("Image could not be decoded.")
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

def parse_yolo(txt):
    boxes=[]; labels=[]
    if not txt.strip(): return boxes,labels
    for n,line in enumerate(txt.strip().splitlines(),1):
        v=line.split()
        if len(v)!=5: raise ValueError(f"YOLO row {n} must contain five values.")
        try: c=int(v[0]); b=list(map(float,v[1:]))
        except ValueError: raise ValueError(f"Non-numeric value in YOLO row {n}.")
        x,y,w,h=b
        if not 0<=c<len(CLASSES): raise ValueError(f"Invalid class ID in row {n}.")
        if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1.000001 and y+h/2<=1.000001):
            raise ValueError(f"Invalid normalized box in row {n}.")
        boxes.append(b); labels.append(c)
    return boxes,labels

def validate(txt):
    r={"rows":0,"valid_rows":0,"malformed_rows":0,"invalid_class_ids":0,"invalid_coordinates":0}
    for line in txt.strip().splitlines():
        if not line.strip(): continue
        r["rows"]+=1; v=line.split()
        if len(v)!=5: r["malformed_rows"]+=1; continue
        try: c=int(v[0]); x,y,w,h=map(float,v[1:])
        except ValueError: r["malformed_rows"]+=1; continue
        if not 0<=c<len(CLASSES): r["invalid_class_ids"]+=1; continue
        if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1.000001 and y+h/2<=1.000001):
            r["invalid_coordinates"]+=1; continue
        r["valid_rows"]+=1
    return r

def valid_yolo_rows(txt):
    """Return valid boxes/classes plus a row-level issue list; invalid rows are skipped safely."""
    boxes=[]; labels=[]; issues=[]
    for n,line in enumerate(txt.splitlines(),1):
        if not line.strip(): continue
        v=line.split()
        if len(v)!=5:
            issues.append({"line":n,"reason":"Expected 5 fields","content":line}); continue
        try: c=int(v[0]); x,y,w,h=map(float,v[1:])
        except ValueError:
            issues.append({"line":n,"reason":"Non-numeric field","content":line}); continue
        if not 0<=c<len(CLASSES):
            issues.append({"line":n,"reason":"Class ID outside configured class range","content":line}); continue
        if not np.isfinite([x,y,w,h]).all() or not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1 and x-w/2>=-1e-6 and y-h/2>=-1e-6 and x+w/2<=1.000001 and y+h/2<=1.000001):
            issues.append({"line":n,"reason":"Invalid normalized coordinates or zero-sized box","content":line}); continue
        boxes.append([x,y,w,h]); labels.append(c)
    return boxes,labels,issues

def clean_yolo_text(txt):
    boxes,labels,issues=valid_yolo_rows(txt)
    return encode(boxes,labels),issues

def encode(boxes,labels):
    return "".join(str(int(c))+" "+" ".join(f"{float(x):.6f}" for x in b)+"\n" for c,b in zip(labels,boxes))

def transform(im,boxes,labels,mode,bright=.22,contrast=.22,angle=7,scale=.08,persp=.10):
    bp=A.BboxParams(format="yolo",label_fields=["class_labels"],min_visibility=.25,clip=True,filter_invalid_bboxes=True)
    if mode=="Lighting":
        t=A.Compose([A.RandomBrightnessContrast(brightness_limit=(-bright,bright),contrast_limit=(-contrast,contrast),p=1),A.RandomGamma(gamma_limit=(70,135),p=1)])
        r=t(image=im); nb,nl=list(boxes),list(labels)
    elif mode=="Camera angle":
        t=A.Compose([A.Affine(rotate=(-angle,angle),scale=(max(.65,1-scale),1+scale),translate_percent=(-.025,.025),p=1,border_mode=cv2.BORDER_REFLECT_101)],bbox_params=bp)
        r=t(image=im,bboxes=boxes,class_labels=labels); nb,nl=list(r["bboxes"]),list(r["class_labels"])
    elif mode=="Perspective":
        ps=min(.15,max(.02,persp))
        t=A.Compose([A.Perspective(scale=(ps*.65,ps),keep_size=True,fit_output=False,p=1,border_mode=cv2.BORDER_REFLECT_101)],bbox_params=bp)
        r=t(image=im,bboxes=boxes,class_labels=labels); nb,nl=list(r["bboxes"]),list(r["class_labels"])
    else:
        ps=min(.15,max(.02,persp))
        t=A.Compose([A.Affine(rotate=(-angle,angle),scale=(max(.65,1-scale),1+scale),translate_percent=(-.02,.02),p=1,border_mode=cv2.BORDER_REFLECT_101),A.Perspective(scale=(ps*.55,ps),keep_size=True,fit_output=False,p=1,border_mode=cv2.BORDER_REFLECT_101),A.RandomBrightnessContrast(brightness_limit=(-bright,bright),contrast_limit=(-contrast,contrast),p=1),A.RandomGamma(gamma_limit=(75,130),p=1)],bbox_params=bp)
        r=t(image=im,bboxes=boxes,class_labels=labels); nb,nl=list(r["bboxes"]),list(r["class_labels"])
    return r["image"],nb,nl

def draw(im,boxes,labels):
    out=im.copy(); h,w=out.shape[:2]; thickness=max(2,round(min(h,w)/500))
    for b,c in zip(boxes,labels):
        x,y,bw,bh=b
        p1=(max(0,int((x-bw/2)*w)),max(0,int((y-bh/2)*h)))
        p2=(min(w-1,int((x+bw/2)*w)),min(h-1,int((y+bh/2)*h)))
        cv2.rectangle(out,p1,p2,(0,255,70),thickness)
        text=CLASSES[int(c)]
        scale=max(.45,min(.8,w/1800))
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
        y0=max(th+8,p1[1]); cv2.rectangle(out,(p1[0],y0-th-7),(min(w-1,p1[0]+tw+8),y0+2),(0,115,35),-1)
        cv2.putText(out,text,(p1[0]+4,y0-3),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
    return out

def png(im):
    b=io.BytesIO(); Image.fromarray(im).convert("RGB").save(b,format="PNG"); return b.getvalue()

def image_paths(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def safe_extract(upload,dest):
    with zipfile.ZipFile(upload) as z:
        for m in z.infolist():
            target=(dest/m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())+os.sep) and target!=dest.resolve(): raise ValueError("Unsafe path in ZIP.")
        z.extractall(dest)

def zip_dir(folder):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(folder))
    return b.getvalue()

def make_tf_source():
    lines = [
        "import os, glob, cv2, numpy as np, tensorflow as tf",
        "ROOT=os.path.dirname(os.path.abspath(__file__)); SIZE=640; BATCH=8",
        "def preprocess(ip,lp):",
        " im=cv2.imread(ip)",
        " if im is None: raise ValueError(ip)",
        " im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); h,w=im.shape[:2]; scale=min(SIZE/w,SIZE/h)",
        " nw,nh=round(w*scale),round(h*scale); resized=cv2.resize(im,(nw,nh)); canvas=np.zeros((SIZE,SIZE,3),np.uint8)",
        " px,py=(SIZE-nw)//2,(SIZE-nh)//2; canvas[py:py+nh,px:px+nw]=resized; boxes=[]",
        " if os.path.exists(lp):",
        "  for line in open(lp,encoding='utf-8'):",
        "   v=line.split()",
        "   if len(v)!=5: continue",
        "   try: c=int(v[0]); x,y,bw,bh=map(float,v[1:])",
        "   except (ValueError,TypeError): continue",
        "   if not (0<=c<6 and np.isfinite([x,y,bw,bh]).all() and 0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1 and x-bw/2>=0 and y-bh/2>=0 and x+bw/2<=1 and y+bh/2<=1): continue",
        "   boxes.append([c,(x*w*scale+px)/SIZE,(y*h*scale+py)/SIZE,bw*w*scale/SIZE,bh*h*scale/SIZE])",
        " return canvas.astype(np.float32)/255.,np.asarray(boxes,np.float32).reshape(-1,5)",
        "def gen(split):",
        " idir=os.path.join(ROOT,split,'images'); ldir=os.path.join(ROOT,split,'labels')",
        " paths=sorted(sum([glob.glob(os.path.join(idir,p)) for p in ('*.jpg','*.jpeg','*.png')],[]))",
        " for p in paths: yield preprocess(p,os.path.join(ldir,os.path.splitext(os.path.basename(p))[0]+'.txt'))",
        "SIG=(tf.TensorSpec((SIZE,SIZE,3),tf.float32),tf.TensorSpec((None,5),tf.float32))",
        "def make_ds(split):",
        " ds=tf.data.Dataset.from_generator(lambda:gen(split),output_signature=SIG)",
        " return ds.padded_batch(BATCH,padded_shapes=([SIZE,SIZE,3],[None,5]),padding_values=(np.float32(0),np.float32(-1))).prefetch(tf.data.AUTOTUNE)",
        "train_dataset=make_ds('train'); val_dataset=make_ds('val'); test_dataset=make_ds('test')",
        "if __name__=='__main__':",
        " for n,d in [('train',train_dataset),('val',val_dataset),('test',test_dataset)]:",
        "  for x,y in d.take(1): print(n,x.shape,y.shape,float(tf.reduce_min(x)),float(tf.reduce_max(x)))",
    ]
    return "\n".join(lines) + "\n"

st.markdown('<div class="brand"><div class="brandmark">▦</div><div><div class="brandtitle">ShelfAugment AI</div><div class="brandsub">Retail image intelligence · Dataset preparation workspace</div></div><div style="margin-left:auto;text-align:right;color:#DDE6ED;font-size:11px">PROJECT WORKSPACE<br><b style="color:#fff">YOLO ANNOTATION AWARE</b></div></div>',unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### ShelfAugment AI"); st.caption("Retail image augmentation")
    page=st.radio("WORKSPACE",["Overview","Image Studio","Batch Processing","Dataset QA & TensorFlow","Project Guide"])
    st.divider(); st.caption("Annotation · YOLO"); st.caption("Product categories · 6"); st.caption("TensorFlow export · 640 × 640")

if page=="Overview":
    st.markdown('<div class="kicker">WORKSPACE OVERVIEW</div>',unsafe_allow_html=True); st.title("Project dashboard")
    st.write("Live activity from image transformations, batch jobs and dataset preparation in this app session.")
    retention=100*st.session_state.generated_boxes/st.session_state.original_boxes if st.session_state.original_boxes else None
    c=st.columns(4)
    with c[0]: card("Images processed",f'{st.session_state.images_processed:,}',"Current session")
    with c[1]: card("Augmented samples",f'{st.session_state.augmented_samples:,}',"Generated outputs")
    with c[2]: card("Bounding-box retention",f'{retention:.1f}%' if retention is not None else "—","Weighted across labelled jobs")
    with c[3]: card("Batch runs",f'{st.session_state.batch_runs:,}',"Completed jobs")
    st.markdown("### Augmentation activity")
    if st.session_state.method_counts:
        df=pd.DataFrame([{"Method":METHODS.get(k,k.title()),"Samples":v} for k,v in st.session_state.method_counts.items()]).set_index("Method")
        a,b=st.columns([1.4,1])
        with a: st.markdown('<div class="panel"><b>Samples generated by method</b>',unsafe_allow_html=True); st.bar_chart(df,color="#526D82",height=260); st.markdown("</div>",unsafe_allow_html=True)
        with b:
            st.markdown('<div class="panel"><b>Annotation summary</b>',unsafe_allow_html=True)
            st.metric("Original boxes",f'{st.session_state.original_boxes:,}'); st.metric("Generated boxes",f'{st.session_state.generated_boxes:,}'); st.metric("Retention",f'{retention:.1f}%' if retention is not None else "—"); st.markdown("</div>",unsafe_allow_html=True)
    else: st.markdown('<div class="panel"><b>No processing activity in this session yet.</b><br><span style="color:#526D82">Figures are calculated from actual actions; no sample values are inserted.</span></div>',unsafe_allow_html=True)
    st.markdown("### Recent activity")
    if st.session_state.history:
        hist=pd.DataFrame(st.session_state.history); st.dataframe(hist,use_container_width=True,hide_index=True)
        st.download_button("Export activity history CSV",hist.to_csv(index=False).encode(),"shelfaugment_activity.csv","text/csv")
    else: st.info("Recent processing activity will appear here.")
    st.markdown("### Workflow")
    cols=st.columns(4)
    for col,n,t,d in zip(cols,["01","02","03","04"],["Upload","Augment","Validate","Export"],["Add shelf images and labels.","Apply controlled transformations.","Check classes and box coordinates.","Download datasets and TF input files."]):
        with col: st.markdown(f'<div class="metriccard"><div class="kicker">{n}</div><b>{t}</b><p style="font-size:12px;color:#526D82">{d}</p></div>',unsafe_allow_html=True)

elif page=="Image Studio":
    st.markdown('<div class="kicker">SINGLE IMAGE WORKFLOW</div>',unsafe_allow_html=True); st.title("Image studio")
    st.write("Apply a controlled transformation while keeping YOLO annotations aligned.")
    left,right=st.columns([.95,1.05],gap="large")
    with left:
        image=st.file_uploader("Shelf image",["jpg","jpeg","png"],key="single_image")
        label=st.file_uploader("YOLO annotation (optional)",["txt"],key="single_label")
        if image: st.caption(f"Selected image: **{image.name}**")
        if label: st.caption(f"Selected annotation: **{label.name}**")
        mode=st.selectbox("Augmentation method",["Lighting","Camera angle","Perspective","Combined"])
        with st.expander("Transformation parameters",expanded=True):
            bright=st.slider("Brightness strength",0.,.6,.28,.01); contrast=st.slider("Contrast strength",0.,.6,.28,.01)
            angle=st.slider("Maximum rotation",0,18,8,1,format="%d°"); scale=st.slider("Scale variation",0.,.30,.10,.01)
            persp=st.slider("Maximum perspective strength",.02,.15,.10,.01,help="Higher values create a stronger viewpoint change; inspect boxes after generation.")
        run=st.button("Generate augmented image",type="primary",use_container_width=True,disabled=image is None)
    if image:
        try:
            im=decode(image); txt=label.getvalue().decode() if label else ""; boxes,labels=parse_yolo(txt)
            with right: st.markdown("#### Original image"); st.image(draw(im,boxes,labels),use_container_width=True); st.caption(f"{im.shape[1]} × {im.shape[0]} px · {len(boxes)} annotated objects")
            if run:
                aug,nb,nl=transform(im,boxes,labels,mode,bright,contrast,angle,scale,persp)
                st.session_state.single_result={"image":png(aug),"preview":png(draw(aug,nb,nl)),"labels":encode(nb,nl),"mode":mode,"original":len(boxes),"generated":len(nb),"filename":Path(image.name).stem}
                st.session_state.images_processed+=1; st.session_state.augmented_samples+=1; st.session_state.original_boxes+=len(boxes); st.session_state.generated_boxes+=len(nb); st.session_state.method_counts[mode.lower().replace(" ","_")]+=1
                add_event({"Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Operation":"Single image","Method":mode,"Images":1,"Samples":1,"Boxes":f"{len(boxes)} → {len(nb)}"})
            if st.session_state.single_result:
                r=st.session_state.single_result; st.markdown("#### Original vs augmented comparison"); a,b,c=st.columns([1,1,1])
                with a: st.image(draw(im,boxes,labels),caption="Original · green YOLO boxes",use_container_width=True)
                with b: st.image(r.get("preview",r["image"]),caption=r["mode"]+" · updated YOLO boxes",use_container_width=True)
                with c:
                    st.metric("Original boxes",r["original"]); st.metric("Retained boxes",r["generated"]); st.metric("Filtered boxes",max(0,r["original"]-r["generated"]))
                    st.download_button("Download augmented image",r["image"],r["filename"]+"_augmented.png","image/png",use_container_width=True)
                    st.download_button("Download updated YOLO labels",r["labels"],r["filename"]+"_augmented.txt","text/plain",use_container_width=True)
        except Exception as e: st.error(f"Unable to process image: {e}")
    else: st.markdown('<div class="softnote"><b>Annotation note:</b> YOLO labels are optional. Without labels, images can be transformed, but object boxes cannot be measured or updated.</div>',unsafe_allow_html=True)

elif page=="Batch Processing":
    st.markdown('<div class="kicker">MULTI-IMAGE WORKFLOW</div>',unsafe_allow_html=True); st.title("Batch processing")
    st.write("Create augmented datasets with updated labels, metadata and input annotation checks.")
    st.markdown('<div class="softnote">ZIP must contain images/ at its root and an optional labels/ folder. Matching basenames are paired automatically.</div>',unsafe_allow_html=True)
    upload=st.file_uploader("Dataset ZIP",["zip"],key="batch_upload")
    if upload: st.caption(f"Selected dataset: **{upload.name}** · {upload.size/1024/1024:.1f} MB")
    selected=st.multiselect("Augmentation methods",list(METHODS),default=list(METHODS),format_func=lambda x:METHODS[x])
    a,b,c=st.columns(3)
    with a: bright=st.slider("Brightness / contrast",0.,.6,.24,.01)
    with b: angle=st.slider("Maximum rotation",0,18,8,1,format="%d°")
    with c: persp=st.slider("Perspective strength",.02,.15,.10,.01)
    run=st.button("Process dataset",type="primary",disabled=upload is None or not selected,use_container_width=True)
    if run and upload:
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); safe_extract(upload,root); idir=root/"images"; ldir=root/"labels"
                if not idir.exists(): raise ValueError("ZIP must contain images/ at its root.")
                paths=image_paths(idir)
                if not paths: raise ValueError("No JPG, JPEG or PNG images found.")
                if len(paths)>1500: raise ValueError("Maximum 1,500 images per batch. Split larger ZIP files.")
                out=root/"output"; out.mkdir(); rows=[]; qa=[]; progress=st.progress(0)
                for i,p in enumerate(paths):
                    im=np.array(Image.open(p).convert("RGB")); lp=ldir/(p.stem+".txt"); txt=lp.read_text(encoding="utf-8") if lp.exists() else ""
                    q=validate(txt); boxes,labels,issues=valid_yolo_rows(txt); qa.append({"image":p.name,"label_found":lp.exists(),**q,"invalid_details":"; ".join(f"line {x['line']}: {x['reason']}" for x in issues)})
                    for m in selected:
                        aug,nb,nl=transform(im,boxes,labels,METHODS[m],bright,bright,angle,.05,persp)
                        oi=out/m/"images"; ol=out/m/"labels"; oi.mkdir(parents=True,exist_ok=True); ol.mkdir(parents=True,exist_ok=True)
                        name=p.stem+"_"+m; Image.fromarray(aug).save(oi/(name+".jpg"),quality=95); (ol/(name+".txt")).write_text(encode(nb,nl))
                        rows.append({"source_image":p.name,"augmentation":m,"original_boxes":len(boxes),"generated_boxes":len(nb),"image":f"{m}/images/{name}.jpg","label":f"{m}/labels/{name}.txt"})
                    progress.progress((i+1)/len(paths))
                df=pd.DataFrame(rows); df.to_csv(out/"augmentation_metadata.csv",index=False); pd.DataFrame(qa).to_csv(out/"annotation_validation.csv",index=False)
                invalid=sum(q["malformed_rows"]+q["invalid_class_ids"]+q["invalid_coordinates"] for q in qa)
                report={"created_at":datetime.now().isoformat(timespec="seconds"),"original_images":len(paths),"methods":selected,"augmented_samples":len(rows),"original_boxes":int(df.original_boxes.sum()),"generated_boxes":int(df.generated_boxes.sum()),"invalid_input_rows":invalid,"images_without_labels":sum(not q["label_found"] for q in qa),"filtered_invalid_rows":invalid}
                (out/"batch_report.json").write_text(json.dumps(report,indent=2))
                zbuf=io.BytesIO()
                with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as z:
                    for p in out.rglob("*"):
                        if p.is_file(): z.write(p,p.relative_to(out))
                st.session_state.batch_df=df; st.session_state.batch_zip=zbuf.getvalue(); st.session_state.batch_report=report
                st.session_state.images_processed+=len(paths); st.session_state.augmented_samples+=len(rows); st.session_state.batch_runs+=1; st.session_state.original_boxes+=report["original_boxes"]; st.session_state.generated_boxes+=report["generated_boxes"]
                for m in selected: st.session_state.method_counts[m]+=len(paths)
                add_event({"Time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Operation":"Batch","Method":", ".join(METHODS[m] for m in selected),"Images":len(paths),"Samples":len(rows),"Boxes":f'{report["original_boxes"]} → {report["generated_boxes"]}'})
                st.success(f'Processed {len(paths)} images and generated {len(rows)} augmented samples.')
        except Exception as e: st.error(f"Batch processing failed: {e}")
    if st.session_state.batch_report:
        r=st.session_state.batch_report; st.markdown("### Latest batch summary")
        a,b,c,d=st.columns(4); a.metric("Images processed",r["original_images"]); b.metric("Augmented samples",r["augmented_samples"]); c.metric("Original boxes",r["original_boxes"]); d.metric("Generated boxes",r["generated_boxes"])
        st.download_button("Download augmented dataset ZIP",st.session_state.batch_zip,"ShelfAugment_AI_batch_output.zip","application/zip",use_container_width=True)
        st.download_button("Download metadata CSV",st.session_state.batch_df.to_csv(index=False).encode(),"augmentation_metadata.csv","text/csv")
        st.markdown("#### Augmentation metadata"); st.dataframe(st.session_state.batch_df,use_container_width=True,hide_index=True)
        st.caption(f'Invalid input annotation rows skipped: {r["invalid_input_rows"]} · Images without labels: {r["images_without_labels"]}')
        with st.expander("View annotation validation report"):
            with zipfile.ZipFile(io.BytesIO(st.session_state.batch_zip)) as _z:
                _qa=pd.read_csv(_z.open("annotation_validation.csv"))
            st.dataframe(_qa,use_container_width=True,hide_index=True)
            st.download_button("Download annotation validation CSV",_qa.to_csv(index=False).encode(),"annotation_validation.csv","text/csv")

elif page=="Dataset QA & TensorFlow":
    st.markdown('<div class="kicker">DATASET QUALITY & MODEL INPUT PREPARATION</div>',unsafe_allow_html=True); st.title("Dataset QA & TensorFlow export")
    st.write("Validate original YOLO annotations and prepare a leakage-aware TensorFlow input package.")
    st.markdown("### 1 · Original dataset quality check")
    qa=st.file_uploader("Original YOLO dataset ZIP",["zip"],key="qa_original")
    if qa: st.caption(f"Selected dataset: **{qa.name}** · {qa.size/1024/1024:.1f} MB")
    if st.button("Validate original dataset",disabled=qa is None):
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); safe_extract(qa,root); rows=[]; details=[]
                if (root/"images"/"train").exists() or (root/"images"/"test").exists():
                    for split in ("train","test","valid","val"):
                        idir=root/"images"/split; ldir=root/"labels"/split
                        if idir.exists():
                            for im in image_paths(idir):
                                lp=ldir/(im.stem+".txt"); txt=lp.read_text(encoding="utf-8",errors="replace") if lp.exists() else ""
                                rows.append({"split":split,"image":im.name,"label_found":lp.exists(),**validate(txt)})
                                _,_,issues=valid_yolo_rows(txt)
                                for issue in issues: details.append({"split":split,"image":im.name,**issue})
                else:
                    idir=root/"images"; ldir=root/"labels"
                    if not idir.exists(): raise ValueError("ZIP needs images/ or images/train and images/test folders.")
                    for im in image_paths(idir):
                        lp=ldir/(im.stem+".txt"); txt=lp.read_text(encoding="utf-8",errors="replace") if lp.exists() else ""
                        rows.append({"split":"unsplit","image":im.name,"label_found":lp.exists(),**validate(txt)})
                        _,_,issues=valid_yolo_rows(txt)
                        for issue in issues: details.append({"split":"unsplit","image":im.name,**issue})
                if not rows: raise ValueError("No images found in the ZIP's images folders.")
                st.session_state.qa_df=pd.DataFrame(rows); st.session_state.qa_details=pd.DataFrame(details)
        except Exception as e: st.error(f"Validation failed: {e}")
    if st.session_state.qa_df is not None:
        q=st.session_state.qa_df; invalid=int(q[["malformed_rows","invalid_class_ids","invalid_coordinates"]].sum().sum())
        a,b,c,d=st.columns(4); a.metric("Images scanned",len(q)); b.metric("Labels found",int(q.label_found.sum())); c.metric("Valid rows",int(q.valid_rows.sum())); d.metric("Invalid rows",invalid)
        st.dataframe(q,use_container_width=True,hide_index=True); st.download_button("Download QA report CSV",q.to_csv(index=False).encode(),"dataset_quality_report.csv","text/csv")
        if st.session_state.get("qa_details") is not None and not st.session_state.qa_details.empty:
            st.warning("Invalid annotation rows were found. These rows are identified below and will be skipped during TensorFlow export.")
            st.dataframe(st.session_state.qa_details,use_container_width=True,hide_index=True)
            st.download_button("Download invalid annotation details",st.session_state.qa_details.to_csv(index=False).encode(),"invalid_annotation_details.csv","text/csv")
        elif st.session_state.get("qa_details") is not None: st.success("No invalid annotation rows found.")
    st.divider(); st.markdown("### 2 · TensorFlow dataset preparation")
    st.write("Original training images are split by identity. Validation and test remain original-only; augmented samples are added to training only.")
    orig=st.file_uploader("Original dataset ZIP (images/train, images/test, labels/train, labels/test)",["zip"],key="tf_original")
    aug=st.file_uploader("Augmented ZIP from Batch Processing (optional)",["zip"],key="tf_augmented")
    if orig: st.caption(f"Original dataset selected: **{orig.name}** · {orig.size/1024/1024:.1f} MB")
    if aug: st.caption(f"Augmented dataset selected: **{aug.name}** · {aug.size/1024/1024:.1f} MB")
    a,b=st.columns(2)
    with a: valfrac=st.slider("Validation fraction",.10,.30,.20,.05)
    with b: seed=st.number_input("Random seed",0,99999,42)
    if st.button("Prepare TensorFlow export package",type="primary",disabled=orig is None):
        try:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); src=root/"src"; src.mkdir(); safe_extract(orig,src)
                req=[src/"images"/"train",src/"images"/"test",src/"labels"/"train",src/"labels"/"test"]
                if not all(p.exists() for p in req): raise ValueError("Original ZIP must contain images/train, images/test, labels/train and labels/test.")
                train=image_paths(src/"images"/"train"); test=image_paths(src/"images"/"test")
                if not train or not test: raise ValueError("Train and test folders both need images.")
                rng=random.Random(int(seed)); sh=train.copy(); rng.shuffle(sh); nv=max(1,round(len(train)*valfrac)) if len(train)>1 else 0
                valstems={p.stem for p in sh[:nv]}; trainstems={p.stem for p in train}-valstems
                out=root/"tensorflow_dataset"
                for s in ("train","val","test"): (out/s/"images").mkdir(parents=True); (out/s/"labels").mkdir()
                filtered_rows=[0]
                def cp(im,split,ls,prefix):
                    name=prefix+im.name; shutil.copy2(im,out/split/"images"/name); lp=src/"labels"/ls/(im.stem+".txt")
                    txt=lp.read_text(encoding="utf-8",errors="replace") if lp.exists() else ""; clean,issues=clean_yolo_text(txt); filtered_rows[0]+=len(issues)
                    (out/split/"labels"/(Path(name).stem+".txt")).write_text(clean,encoding="utf-8")
                for im in train: cp(im,"val" if im.stem in valstems else "train","train","original_")
                for im in test: cp(im,"test","test","test_")
                added=0
                if aug:
                    ar=root/"aug"; ar.mkdir(); safe_extract(aug,ar); mp=ar/"augmentation_metadata.csv"
                    if not mp.exists(): raise ValueError("Augmented ZIP needs augmentation_metadata.csv.")
                    df=pd.read_csv(mp)
                    if not {"source_image","image","label"}.issubset(df.columns): raise ValueError("Metadata missing source_image, image or label.")
                    for _,row in df.iterrows():
                        if Path(str(row.source_image)).stem not in trainstems: continue
                        rel_image=Path(str(row.image)); rel_label=Path(str(row.label))
                        if rel_image.is_absolute() or ".." in rel_image.parts or rel_label.is_absolute() or ".." in rel_label.parts: continue
                        ip=ar/rel_image; lp=ar/rel_label
                        if not ip.exists(): continue
                        name="aug_"+Path(str(row.image)).parts[0]+"_"+ip.name; shutil.copy2(ip,out/"train"/"images"/name)
                        txt=lp.read_text(encoding="utf-8",errors="replace") if lp.exists() else ""; clean,issues=clean_yolo_text(txt); filtered_rows[0]+=len(issues)
                        (out/"train"/"labels"/(Path(name).stem+".txt")).write_text(clean,encoding="utf-8")
                        added+=1
                counts={s:{"images":len(image_paths(out/s/"images")),"labels":len(list((out/s/"labels").glob("*.txt")))} for s in ("train","val","test")}
                report={"framework":"TensorFlow","image_size":640,"batch_size":8,"normalization":"RGB / 255.0","annotation_format":"YOLO","seed":int(seed),"validation_fraction":valfrac,"augmented_samples_added_to_train":added,"invalid_annotation_rows_filtered":filtered_rows[0],"splits":counts,"leakage_control":"Validation and test contain original images only; augmented samples are added to training only."}
                (out/"tensorflow_integration_report.json").write_text(json.dumps(report,indent=2))
                (out/"tensorflow_input_pipeline.py").write_text(make_tf_source())
                (out/"README_TENSORFLOW.md").write_text("Install tensorflow, opencv-python-headless and numpy in Kaggle/Colab. Run python tensorflow_input_pipeline.py. Images are letterboxed to 640x640 RGB and normalized to 0-1. YOLO boxes are [class_id, x_center, y_center, width, height]. Padded rows use -1. This prepares input datasets; it does not train a detector.")
                zbuf=io.BytesIO()
                with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as z:
                    for p in out.rglob("*"):
                        if p.is_file(): z.write(p,p.relative_to(out))
                st.session_state.tf_zip=zbuf.getvalue(); st.session_state.tf_report=report; st.success("TensorFlow preparation package created.")
        except Exception as e: st.error(f"TensorFlow export failed: {e}")
    if st.session_state.tf_report:
        r=st.session_state.tf_report; a,b,c,d=st.columns(4)
        card("Training images",r["splits"]["train"]["images"],"Original train + augmented"); card("Validation images",r["splits"]["val"]["images"],"Original only"); card("Test images",r["splits"]["test"]["images"],"Original only"); card("Augmented train samples",r["augmented_samples_added_to_train"],"Added to training only")
        st.info(r["leakage_control"]); st.caption(f"Invalid annotation rows filtered during export: {r.get('invalid_annotation_rows_filtered',0)}")
        st.download_button("Download TensorFlow dataset package",st.session_state.tf_zip,"ShelfAugment_AI_TensorFlow_Dataset.zip","application/zip",use_container_width=True)
        st.download_button("Download TensorFlow report JSON",json.dumps(r,indent=2),"tensorflow_integration_report.json","application/json")

else:
    st.markdown('<div class="kicker">PROJECT DOCUMENTATION</div>',unsafe_allow_html=True); st.title("Project guide")
    st.write("ShelfAugment AI creates controlled retail shelf-image variations and corresponding YOLO annotations for downstream model development.")
    st.markdown("### Product class mapping"); st.dataframe(pd.DataFrame({"Class ID":range(6),"Product":CLASSES}),use_container_width=True,hide_index=True)
    st.markdown("### Transformation methods"); st.dataframe(pd.DataFrame([["Lighting","Brightness, contrast and gamma","Original boxes retained"],["Camera angle","Affine rotation and scale","YOLO boxes transformed"],["Perspective","Viewpoint variation","Boxes transformed and visibility-filtered"],["Combined","Geometry and lighting","YOLO boxes transformed"]],columns=["Method","Operation","Annotation handling"]),use_container_width=True,hide_index=True)
    st.markdown("### Annotation format"); st.code("class_id x_center y_center width height")
    st.write("Coordinates are normalized to 0–1. Geometric transformations update box coordinates and filter low-visibility boxes.")
    st.markdown("### TensorFlow preparation"); st.write("The export kit prepares 640 × 640 RGB letterboxed images, normalized pixels, YOLO box arrays and padded tf.data batches. It does not train or evaluate a detector.")
    st.info("Planogram-compliance scoring and object-detector training are downstream tasks outside the current application's scope.")
