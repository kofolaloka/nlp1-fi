import gzip
import pandas as pd
import os

class Triple(dict):
    
    def __init__(self,*args):
        super(Triple,self).__init__()
        assert len(args) == len(Triple.members())
        
        for h,a in zip(Triple.members(),args):
            self[h] = a

    @staticmethod
    def members():
        return ['v','s','o','value']

    def totabs(self):
        return '\t'.join(map(str,self.tolist()))

    def tolist(self):
        return [self[m] for m in Triple.members()]

    def __str__(self):
        return super(Triple,self).__str__()

    def __repr__(self):
        return 'Triple('+','.join(map(str,self.tolist()))+')'

class Tome(object):
    
    def __init__(self, filename):
        self.filename = filename
    
    def _unbox_as_df(self):

        if not os.path.isfile(self.filename):
            raise Exception("file %s does not exist"%self.filename)
        handle = gzip.open(self.filename, 'rb')
        df = pd.DataFrame.from_csv(
            handle,
            sep='\t',
            #names=Triple.members(),
            header=None,
            index_col=None
        )
        return df
    
    def _group(self, fields=None):

        df = self._unbox_as_df()

        if fields is None:
            fields = Triple.members()
        
        field_idx = map(Triple.members().index,fields)
        ret = df.groupby(field_idx)
        return ret
    
    def _group_sum_df(self,fields):
        tmp = self._group(fields)
        ret = tmp.sum().reset_index()
        return ret
    
    def group_sum(self, fields):
        tmp = self._group_sum_df(fields)
        for i,row in tmp.iterrows():
            yield Triple(*row)
    
    def writer(self):
        handle = gzip.open(self.filename, "wb")

        def _fn(payload):
            if type(payload) is Triple:
                buf = payload.totabs()+'\n'
            else:
                buf = payload
            handle.write(buf)
            handle.flush()
        
        return _fn

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Tome("%s")'%self.filename

