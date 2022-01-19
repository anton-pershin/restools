from operator import itemgetter

from thequickmath.misc import index_for_almost_exact_coincidence


class Function:
    """
    Class Function is a simple representation of any abstract data defined on some domain so that there is a one-to-one
    correspondence between a given element of data and some element of the domain. Moreover, it is assumed that the
    domain constitutes a set with strict partial order (it is important because both data and domain are sorted in the
    constructor). The class has only two properties: domain and data. Using the methods of the class, one can access
    the value of the function at a certain point of the domain.
    """
    def __init__(self, data, domain):
        domain_and_data_sorted = sorted(zip(domain, data), key=itemgetter(0))
        self._domain = [pair[0] for pair in domain_and_data_sorted]
        self._data = [pair[1] for pair in domain_and_data_sorted]

    @property
    def domain(self):
        return self._domain

    @property
    def data(self):
        return self._data

    def at(self, x):
        """
        Returns an element of the data associated with x

        :param x: an element of the domain
        :return: an element of the data associated with x
        """
        i = index_for_almost_exact_coincidence(self._domain, x)
        return self._data[i]

    def local_maxima(self):
        pass


'''def build_w_local_maxima_approx(Res, datas):
    Nt = -1
    slopes = []
    ks = []
    bs = []
    maxima_number = 1000
    #for task, Re in zip(tasks, Res):
    for Re, data in zip(Res, datas):
        #data = get_integration_data(os.path.join(res.get_task_path(task), 'data-{}'.format(Re)))

        ### HERE WAS SOME POINCARE CODE
#        v_init = data['L2v'][0]
#        P_sequence = []
#        maxima = []
#        T_at_maxima = []
#        for i in range(len(data['L2v'])):
#            if i != 0:
#                v_i = data['L2v'][i - 1]
#                v_ii = data['L2v'][i]
#                if (v_i - v_init) * (v_ii - v_init) < 0:
#                    w_i = data['L2w'][i - 1]
#                    w_ii = data['L2w'][i]
#                    P_sequence.append((w_ii - w_i) / (v_ii - v_i) * v_init + (w_i*v_ii - w_ii*v_i) / (v_ii - v_i))
#
        maxima_indices = local_maxima_indices(data['L2w'][:Nt], threshold=-1)
        T_at_maxima = np.array(data['T'])[maxima_indices]
        maxima = np.array(data['L2w'])[maxima_indices]
        if maxima_number + 1 > len(maxima):
            maxima_number = len(maxima) - 1
        slopes.append(maxima[0])

        ### LOGIC ###
        # To make the task well-conditioned, we count the number of maxima taken into consideration
        # such that it is monotonically decreasing with Re.   drop all the terms which are We wish to consider only positive slopes in the log-scale, so we look for the first n
        if maxima_number > 1:
            if len(np.where(np.array(maxima[:maxima_number]) - data['L2w'][0] <= 0.)[0]) != 0:
                print('Re = {}'.format(Re))
                print(np.array(maxima[:maxima_number]) - data['L2w'][0])
            k, b = np.polyfit(T_at_maxima[:maxima_number], np.log(np.array(maxima[:maxima_number]) - data['L2w'][0]), 1)
        else:
            k = None
            b = 0
        ks.append(k)
        bs.append(b)
    return slopes, ks, bs
'''