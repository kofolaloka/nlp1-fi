import utils
import triple
import numpy as np
A=3 # number of args (v,s,o)
F=200
num_epochs=2000
def initialize_theta():
    return np.ones(F)/F # uniform distribution

def initialize_phi(D):
    r = np.random.random((F,A,D))
    normalizer = np.sum(r,axis=2,keepdims=True) # sum_w phi^a(w) = 1
    normalizer = np.tile(normalizer,(1,1,D))
    phi = r / normalizer
    return phi

def e_step(word_indexes, theta, phi):
    selected = phi[:,range(A),word_indexes] # shape (F,#triples,A)
    pp = np.prod(selected,axis=2) # shape (F,#triples)
    mus = np.expand_dims(theta,1) * pp # shape (F,#triples)
    denom = np.sum(mus, axis=0)
    mus = mus/denom # normalize to sum_f mu_i(f) = 1
    return mus

def m_step_phi(D, word_indexes, mus_times_counts):
    wi = np.array(word_indexes)
    phi = np.zeros((F,A,D))
    selected = phi[:,range(A),word_indexes] # shape (F,#triples,A)
    for a in range(A):
        for curr in range(D):
            ivalues = np.nonzero(wi[:,a] == curr)[0]
            phi[:,a,curr] = np.sum(mus_times_counts[:,ivalues],axis=1)

    # normalization
    denom = np.expand_dims(np.sum(phi, axis=2),2)
    phi = phi/denom
    return phi

def m_step_theta(word_indexes, mus_times_counts):
    propto = np.sum(mus_times_counts, axis=1)
    denom = np.sum(propto)
    theta = propto/denom
    return theta

def m_step(D, word_indexes, counts, mus):
    mus_times_counts = mus * np.array(counts)
    theta = m_step_theta(word_indexes, mus_times_counts)
    phi = m_step_phi(D, word_indexes, mus_times_counts)
    return theta, phi

def loglikelihood(word_indexes, mus, theta, phi):
    term1 = np.sum(np.dot(np.log(theta),mus))

    phi_selected = np.log(phi[:,range(A),word_indexes]+0.0000000000000000001) # shape (F,#triples,A)
    phi_sum = np.sum(phi_selected, axis=2)
    term2 = np.sum(phi_sum*mus)
    return term1 + term2
    """
    for mu in mus.T:
    """

def em(tv):
    D = len(tv.vocabulary)
    theta = initialize_theta()
    phi = initialize_phi(D)
    for epoch in range(num_epochs):
        print "epoch",epoch
        mus = e_step(tv.indexes, theta, phi)
        counts = tv.counts
        theta, phi = m_step(D, tv.indexes, counts, mus)
        #print "theta",theta
        print "loglikelihood",loglikelihood(tv.indexes, mus,theta,phi)
    return mus

def prepare_tomes(in_d):

    tomes = [
        triple.Tome(filename)
        for filename
        in utils.filenames(in_d)
    ]

    print "number of tomes found: %d"%len(tomes)

    tv = triple.TomeVoc(tomes)
    word_indexes = tv.indexes
    return tv

def write_results(tv, mus, out_d):
    frames = np.argmax(mus,axis=0)
    filename_out = utils.new_filename(out_d, "framed_triples.gz")
    tome_out = triple.Tome(filename_out)
    writer = tome_out.writer()
    for tr,frame in zip(tv.triples,frames):
        tr_new = triple.Triple(*(tr.tolist()[:3]+[frame]))
        writer(tr_new)

def main():
    in_d, out_d,m  = utils.argsdirs("Expectation Maximization (Model 0)")
    tv = prepare_tomes(in_d)
    mus = em(tv)
    write_results(tv, mus, out_d)
    import ipdb; ipdb.set_trace()

if __name__=="__main__":
    main()

