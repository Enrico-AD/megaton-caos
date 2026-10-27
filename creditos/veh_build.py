import numpy as np, json, struct, colorsys
from kload import load
MODELS=[('sedan',1),('sedan-sports',1),('hatchback-sports',1),('suv',1),('suv-luxury',1),('van',1),('truck',1),('taxi',0),('delivery',0),('ambulance',0),('firetruck',0),('garbage-truck',0),('police',0)]
S=np.array([1.35,1.42,1.72]) # x(largura),y,z(comprimento) no espaço Kenney
def hsv(c):return colorsys.rgb_to_hsv(*c)
def wheel(cx,cy,cz,r,w,seg=10):
  P=[];N=[];C=[];I=[];TIRE=(0.13,0.13,0.15);HUB=(0.62,0.63,0.68)
  def add(p,n,c):P.append(p);N.append(n);C.append(c);return len(P)-1
  for i in range(seg):
    a0=2*np.pi*i/seg;a1=2*np.pi*(i+1)/seg
    for a in (a0,a1):pass
    y0,z0=np.cos(a0)*r,np.sin(a0)*r;y1,z1=np.cos(a1)*r,np.sin(a1)*r;am=(a0+a1)/2
    n=(0,np.cos(am),np.sin(am))
    q=[add((cx-w/2,cy+y0,cz+z0),n,TIRE),add((cx+w/2,cy+y0,cz+z0),n,TIRE),add((cx+w/2,cy+y1,cz+z1),n,TIRE),add((cx-w/2,cy+y1,cz+z1),n,TIRE)]
    I+= [q[0],q[2],q[1],q[0],q[3],q[2]]
    for sx in (-1,1):
      x=cx+sx*w/2;nn=(sx,0,0);rr=r*0.62
      c0=add((x,cy,cz),nn,HUB);a=add((x,cy+np.cos(a0)*rr,cz+np.sin(a0)*rr),nn,HUB);b=add((x,cy+np.cos(a1)*rr,cz+np.sin(a1)*rr),nn,HUB)
      I+= [c0,a,b] if sx>0 else [c0,b,a]
      a2=add((x,cy+y0,cz+z0),nn,TIRE);b2=add((x,cy+y1,cz+z1),nn,TIRE);a3=add((x,cy+np.cos(a0)*rr,cz+np.sin(a0)*rr),nn,TIRE);b3=add((x,cy+np.cos(a1)*rr,cz+np.sin(a1)*rr),nn,TIRE)
      I+= [a3,a2,b2,a3,b2,b3] if sx>0 else [a3,b2,a2,a3,b3,b2]
  return np.array(P,np.float32),np.array(N,np.float32),np.array(C,np.float32),np.array(I,np.int64)
out=[];blob=b''
for nm,paintable in MODELS:
  ps=load(nm)
  BP=[];BN=[];BC=[];BI=[];paintMask=[]
  wheels=[]
  for p in ps:
    if p['name'].startswith('wheel'):
      P=p['P'];wheels.append((P.min(0)+P.max(0))/2*0+np.array([0,0,0]))  # placeholder
      c=(P.max(0)+P.min(0))/2;r=(P[:,1].max()-P[:,1].min())/2;w=P[:,0].max()-P[:,0].min()
      wheels[-1]=(c,r,w);continue
    o=sum(len(x) for x in BP);BP.append(p['P']);BN.append(p['N']);BC.append(p['C']);BI.append(p['I']+o)
  P=np.concatenate(BP);N=np.concatenate(BN);C=np.concatenate(BC);I=np.concatenate(BI)
  tri=I.reshape(-1,3)
  # pintura: cor saturada com o matiz dominante
  base=None
  if True:
    ar=np.linalg.norm(np.cross(P[tri[:,1]]-P[tri[:,0]],P[tri[:,2]]-P[tri[:,0]]),axis=1)
    sat=np.array([hsv(c)[1] for c in C]);hue=np.array([hsv(c)[0] for c in C])
    tsat=sat[tri[:,0]]>0.3
    hb=np.zeros(36)
    for t,a in zip(tri[tsat],ar[tsat]):hb[int(hue[t[0]]*36)%36]+=a
    h0=(np.argmax(hb)+0.5)/36
    dh=np.minimum(abs(hue-h0),1-abs(hue-h0))
    pv=(sat>0.3)&(dh<0.06)
    tp=pv[tri[:,0]]
    base=np.median(C[pv],axis=0)
  rep=base
  if not paintable: tp=np.zeros(len(tri),bool);base=None
  # luzes: branco-amarelado muito claro e vermelho vivo ganham brilho (glow à noite)
  # transforma: Kenney +z frente? decide pelo lado da grade/luzes claras
  bright=(C.min(1)>0.85)
  zf=P[bright,2].mean() if bright.any() else 1
  front=1  # Kenney: frente em +z (lanternas vermelhas ficam em -z)
  def xf(P,N):
    Q=P*S;Q=np.stack([Q[:,2]*front,Q[:,1],-Q[:,0]*front],1)  # z->x (frente +x)
    M=N/S;M=M/np.linalg.norm(M,axis=1,keepdims=True);M=np.stack([M[:,2]*front,M[:,1],-M[:,0]*front],1);return Q.astype(np.float32),M.astype(np.float32)
  def pack(P,N,C,I):
    # reindexa só os vértices usados
    u,inv=np.unique(I,return_inverse=True);P=P[u];N=N[u];C=C[u];I=inv.astype(np.uint16)
    Q,M=xf(P,N)
    cb=np.clip(C*160,0,255).astype(np.uint8)  # 200 = 1.0 (deixa folga para brilho >1)
    return (Q*1000).round().astype(np.int16),(M*127).round().astype(np.int8),cb,I
  parts={}
  ti_p=tri[tp].reshape(-1);ti_d=tri[~tp].reshape(-1)
  Cd=C.copy()
  lightmask=bright|((C[:,0]>0.85)&(C[:,1]<0.35)&(C[:,2]<0.35))
  Cd[lightmask]*=1.6
  # rodas no det
  WP=[];WN=[];WC=[];WI=[];
  dP,dN,dC,dI=P,N,Cd,ti_d
  for c,r,w in wheels:
    wp,wn,wc,wi=wheel(c[0],c[1],c[2],r,w);o=len(dP);dP=np.concatenate([dP,wp]);dN=np.concatenate([dN,wn]);dC=np.concatenate([dC,wc]);dI=np.concatenate([dI,wi+o])
  ent={'rep':[round(float(x),3) for x in rep],'nm':nm,'paint':[round(float(x),3) for x in base] if base is not None else None}
  for key,(a,b,c,d) in (('d',(dP,dN,dC,dI)),('p',(P,N,(C/base if base is not None else C),ti_p))):
    if len(d)==0:ent[key]=None;continue
    Q,M,cb,II=pack(a,b,c,d)
    e={'nv':len(Q),'ni':len(II)}
    for arr,k in ((Q,'pos'),(M,'nor'),(cb,'col'),(II,'idx')):
      while len(blob)%4:blob+=b'\0'
      e[k]=len(blob);blob+=arr.tobytes()
    ent[key]=e
  Pt,_=xf(P,N);ent['box']=[[round(float(v),2) for v in Pt.min(0)],[round(float(v),2) for v in Pt.max(0)]]
  ent['tris']=int((len(dI)+len(ti_p))//3)
  out.append(ent)
hdr=json.dumps(out,separators=(',',':')).encode()
while (len(hdr)+4)%4:hdr+=b' '
open('veh.bin','wb').write(struct.pack('<I',len(hdr))+hdr+blob)
for e in out:print(e['nm'],e['tris'],e['paint'],e['box'])
import os;print('bytes',os.path.getsize('veh.bin'))
