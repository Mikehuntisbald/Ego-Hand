"""Read an official public ZIP by byte ranges with a pinned expired leaf cert."""
import io, ssl, http.client, hashlib, time

HOST='fouheylab.eecs.umich.edu'
PATH='/~dandans/projects/100DOH/downloads/pascal_voc_format.zip'
LEAF='16a476b7da66f3ab203f941b027da47e2c52a7946c994c58a904f2ef479f2caf'
SIZE=9560896558

def fetch(start,end):
    for attempt in range(4):
        try:
            c=http.client.HTTPSConnection(HOST,timeout=90,context=ssl._create_unverified_context());c.connect()
            assert hashlib.sha256(c.sock.getpeercert(binary_form=True)).hexdigest()==LEAF, 'Public source leaf changed'
            c.request('GET',PATH,headers={'Range':f'bytes={start}-{end}'})
            r=c.getresponse();assert r.status==206 and r.getheader('Content-Range')==f'bytes {start}-{end}/{SIZE}'
            data=r.read();c.close();assert len(data)==end-start+1;return data
        except Exception:
            if attempt==3:raise
            time.sleep(1+attempt)

class RemoteZip(io.RawIOBase):
    def __init__(self):self.pos=0;self.cache={};self.size=SIZE
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        start=self.pos;end=start+n-1
        for (a,b),data in self.cache.items():
            if a<=start and end<=b:self.pos+=n;return data[start-a:end-a+1]
        data=fetch(start,end);self.pos+=n
        if n>65536 or start>self.size-65536:self.cache[(start,end)]=data
        return data
