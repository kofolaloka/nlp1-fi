import gzip
import pandas as pd
import os
import operator
import sklearn.feature_extraction
import os.path

class Triple(dict):

    def __init__(self,*args):
        super(Triple,self).__init__()
        assert len(args) == len(Triple.members()),"Cannot create Triple: given len(args) is %d while len(Triple.members()) is %d"%(len(args),len(Triple.members()))

        for h,a in zip(Triple.members(),args):
            self[h] = a

    @staticmethod
    def members():
        return ['v','s','o','value']

    @staticmethod
    def members_idx(m):
        return map(Triple.members().index,m)

    def totabs(self):
        return '\t'.join(map(str,self.tolist()))

    def tolist(self):
        return [self[m] for m in Triple.members()]

    def __str__(self):
        return super(Triple,self).__str__()

    def __repr__(self):
        return 'Triple('+','.join(map(str,self.tolist()))+')'

class TomeException(Exception):
    pass

class Tome(object):

    def __init__(self, a):
        self.filename = self._df = None
        if type(a) is str:
            self.filename = a
        elif type(a) is pd.DataFrame:
            self._df = a
        elif type(a) is list:
            if all(map(lambda x: type(x) is pd.DataFrame, a)):
                self._df = pd.concat(a)
            elif all(map(lambda x: type(x) is Tome, a)):
                dfs = map(operator.methodcaller('df'), a)
                self._df = pd.concat(dfs)
            else:
                raise TomeException("cannot understand constructor argument (list)")
        else:
            raise TomeException("wrong constructor argument type")

    def _unbox_as_df(self):
        print "unboxing %s ..."%self.filename
        if not os.path.isfile(self.filename):
            raise TomeException("file %s does not exist"%self.filename)

        if os.path.splitext(self.filename)[-1] == '.gz':
            openfunc = gzip.open
        else:
            openfunc = open

        handle = openfunc(self.filename, 'rb')

        df = pd.DataFrame.from_csv(
            handle,
            sep='\t',
            #names=Triple.members(),
            header=None,
            index_col=None
        )
        return df

    def df(self):
        if self.filename is not None:
            return self._unbox_as_df()
        else:
            assert type(self._df) is pd.DataFrame
            return self._df

    def _group(self, members_selected=None):
        df_ = self.df()

        if members_selected is None:
            members_selected = Triple.members()

        field_idx = Triple.members_idx(members_selected)
        ret = df_.groupby(field_idx)
        return ret

    def _group_sum_df(self,fields):
        tmp = self._group(fields)
        ret = tmp.sum().reset_index()
        return ret

    def _to_triples(self,df_):
        for i,row in df_.iterrows():
            yield Triple(*row)

    def __iter__(self):
        return self._to_triples(self.df())

    def group_sum(self, fields):
        tmp = self._group_sum_df(fields)
        return Tome(tmp)

    def sort(self, fields=None,ascending=False):
        df_ = self.df()

        if fields is None:
            fields = ['value']

        field_idx = Triple.members_idx(fields)

        df_sort = df_.sort(field_idx,ascending=ascending)
        return Tome(df_sort)

    def first(self,amount):
        df_ = self.df()
        return Tome(df_[:amount])

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

class TomeVoc(object):
    """
    this object contains auxiliary functions to use in handling a number
    of tomes.
    """

    def __init__(self, tomes):
        assert type(tomes) is list
        self.tomes = tomes

    @property
    def triples(self):
        """
        returns a generator (not a list!, but still iterable) of all the triples
        you can use it for a for loop
        """
        for tome in self.tomes:
            for triple in tome:
                yield triple

    @property
    def vocabulary(self):
        """
        returns the vocabulary of sorted words-strings
        """
        if hasattr(self,'_vocabulary'):
            # lazy evaluation for the win!
            return self._vocabulary

        s = set()
        for triple in self.triples:
            s.update(triple.tolist()[:3])
        tmp = list(s)
        self._vocabulary = sorted(tmp)
        return self._vocabulary

    @property
    def vectors(self):
        raise("not implemented")

    @property
    def indexes(self):
        """
        return all the word indexes in each triple
        (not numpy array, but you just need to np.array' the return value)
        """
        if hasattr(self,'_indexes'):
            # lazy evaluation for the win!
            return self._indexes

        self._indexes = []
        voc = self.vocabulary
        for triple in self.triples:
            triple_indexes = [
                # get the index of the word in the vocabulary for each word-string
                voc.index(word)
                for word
                in triple.tolist()[:3]
            ]
            self._indexes.append(triple_indexes)
        return self._indexes

    @property
    def counts(self):
        """
        returns the counts associated to each triple
        """
        if hasattr(self, '_counts'):
            return self._counts # lazy evaluation
        self._counts = []
        for tome in self.tomes:
            for triple in tome:
                # index of the count is 3 (fourth column)
                self._counts.append(triple.tolist()[3])
        return self._counts

