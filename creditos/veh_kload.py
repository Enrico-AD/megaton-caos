from pygltflib import GLTF2
import numpy as np, struct, sys, io
from PIL import Image
def acc(g,i,blob):
  a=g.accessors[i];bv=g.bufferViews[a.bufferView];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a.type]
  dt={5126:np.float32,5123:np.uint16,5125:np.uint32,5121:np.uint8}[a.componentType]
  off=(bv.byteOffset or 0)+(a.byteOffset or 0);stride=bv.byteStride
  if stride and stride!=n*np.dtype(dt).itemsize:
    raw=np.frombuffer(blob,dtype=np.uint8,count=stride*a.count,offset=off).reshape(a.count,stride)[:,:n*np.dtype(dt).itemsize]
    return np.frombuffer(raw.tobytes(),dtype=dt).reshape(a.count,n)
  return np.frombuffer(blob,dtype=dt,count=a.count*n,offset=off).reshape(a.count,n) if n>1 else np.frombuffer(blob,dtype=dt,count=a.count,offset=off)
def load(name):
  g=GLTF2().load(name+'.glb');blob=g.binary_blob()
  im=g.images[0];bv=g.bufferViews[im.bufferView];img=Image.open(io.BytesIO(blob[bv.byteOffset:bv.byteOffset+bv.byteLength])).convert('RGB');tex=np.asarray(img).astype(np.float32)/255
  parts=[]
  for n in g.nodes:
    if n.mesh is None:continue
    T=np.array(n.translation or [0,0,0]);
    for p in g.meshes[n.mesh].primitives:
      P=acc(g,p.attributes.POSITION,blob).astype(np.float32)+T;N=acc(g,p.attributes.NORMAL,blob).astype(np.float32);UV=acc(g,p.attributes.TEXCOORD_0,blob);I=acc(g,p.indices,blob).astype(np.int64)
      h,w=tex.shape[:2];u=np.clip((UV[:,0]%1)*w,0,w-1).astype(int);v=np.clip((UV[:,1]%1)*h,0,h-1).astype(int);C=tex[v,u]
      parts.append(dict(name=n.name,P=P,N=N,C=C,I=I,rot=n.rotation,sc=n.scale))
  return parts
if __name__=='__main__':
  for nm in sys.argv[1:]:
    ps=load(nm);print('==',nm,[(p['name'],len(p['I'])//3,p['rot'],p['sc']) for p in ps])
    b=[p for p in ps if p['name']=='body'][0];P,I,C=b['P'],b['I'],b['C']
    tri=I.reshape(-1,3);a=np.linalg.norm(np.cross(P[tri[:,1]]-P[tri[:,0]],P[tri[:,2]]-P[tri[:,0]]),axis=1)/2
    cols={};
    for t,ar in zip(tri,a):k=tuple((C[t[0]]*255).round().astype(int));cols[k]=cols.get(k,0)+ar
    for k,v in sorted(cols.items(),key=lambda x:-x[1])[:8]:print('  ',k,round(v,2))
    print('  bbox',P.min(0).round(2),P.max(0).round(2))
