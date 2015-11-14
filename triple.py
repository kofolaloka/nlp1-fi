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

